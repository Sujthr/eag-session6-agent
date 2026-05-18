"""
Decision layer — chooses the NEXT SINGLE action from typed perception + memory.

Routes through LLM Gateway V3 with auto_route="decision" so the gateway
picks a capable LARGE-tier model for this reasoning-heavy task.

Input  → DecisionInput  (typed)
Output → DecisionOutput (typed, Pydantic-validated)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(
    0,
    str(
        Path(__file__).parent
        / "c189e31f-c189-404a-95d8-0ba8b0756c9e"
        / "llm_gatewayV3"
    ),
)
from client import LLM  # noqa: E402

from schemas import DecisionInput, DecisionOutput, MemoryStoreRequest, ToolCallSpec

_llm = LLM()

_TOOL_DOCS: dict[str, str] = {
    "web_search": 'web_search(query: str, max_results: int=3) → list[{title,url,snippet}]',
    "fetch_url": 'fetch_url(url: str) → {status, text}',
    "get_time": 'get_time(timezone: str) → {iso, human, timezone, offset_hours}   — timezone must be IANA, e.g. "Europe/Berlin"',
    "currency_convert": "currency_convert(amount: float, from_currency: str, to_currency: str) → {converted, rate, date}   — use ISO-4217 codes",
    "read_file": 'read_file(path: str) → {content, size_bytes}',
    "list_dir": 'list_dir(path: str=".") → list[{name, type, size_bytes}]',
    "create_file": 'create_file(path: str, content: str) → {ok, path}   — errors if file already exists',
    "update_file": 'update_file(path: str, content: str) → {ok}   — overwrites existing file',
    "edit_file": 'edit_file(path: str, find: str, replace: str) → {ok}',
}

_SYSTEM = """\
You are the Decision module of a multi-layer AI agent.
Your job: pick the SINGLE best action to make progress toward the goal.

Output ONLY valid JSON — no markdown, no prose.

Required JSON structure:
{
  "action": "tool_call" | "store_memory" | "final_answer",
  "tool_call": {"tool": "<name>", "args": {<kwargs>}, "reason": "<why>"},
  "memory_store": {"key": "<snake_case_key>", "value": "<string>", "namespace": "global"},
  "answer": "<complete answer to the user>",
  "reasoning": "<brief chain-of-thought>",
  "confidence": <0.0–1.0>
}

Field rules:
  • Populate only the field relevant to the chosen action (tool_call, memory_store, OR answer).
  • For store_memory: ALSO populate "answer" with a short confirmation to show the user.
  • For final_answer: "answer" must be a complete, user-facing reply — not JSON.
  • reasoning and confidence are always required.
  • If a previous tool call failed, try a different approach or tool.
  • NEVER repeat the exact same tool call (same tool + same args) as a prior iteration.
  • If iteration >= max_iterations-1, use final_answer with the best available information.
  • For currency queries: extract the "converted" value from the previous result and use it as the next call's amount.
  • For file queries: if create_file fails with "already exists", use update_file instead."""


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start <= end:
        text = text[start : end + 1]
    return json.loads(text)


def decide(inp: DecisionInput) -> DecisionOutput:
    tool_docs = "\n".join(
        f"  {doc}"
        for tool, doc in _TOOL_DOCS.items()
        if tool in inp.available_tools
    )

    memory_str = ""
    if inp.memory.facts:
        memory_str = "\nRelevant memories:\n" + "\n".join(
            f"  {f.key}: {f.value}" for f in inp.memory.facts
        )

    results_str = ""
    if inp.tool_results:
        recent = inp.tool_results[-4:]
        lines = []
        for r in recent:
            out = r.get("output", "")
            err = r.get("error", "")
            snippet = json.dumps(out)[:500] if out is not None else f"ERROR: {err}"
            lines.append(f"  [{r.get('tool','')}] → {snippet}")
        results_str = "\nTool results so far:\n" + "\n".join(lines)

    prompt = (
        f"Goal: {inp.goal}\n"
        f"Iteration: {inp.iteration}/{inp.max_iterations}\n"
        f"\nPerception summary:\n"
        f"  Intent : {inp.perception.intent}\n"
        f"  Focus  : {inp.perception.focus}\n"
        f"  Answerable now: {inp.perception.is_answerable}"
        f"{memory_str}"
        f"{results_str}\n"
        f"\nAvailable tools:\n{tool_docs}\n"
        "\nOutput JSON:"
    )

    resp = _llm.chat(
        prompt=prompt,
        system=_SYSTEM,
        auto_route="decision",
        response_format={"type": "json_object"},
        max_tokens=768,
        temperature=0.2,
    )

    raw = resp.get("text", "")
    data: dict[str, Any] = resp.get("parsed") or _parse_json(raw)

    action = data.get("action", "final_answer")

    tool_call = None
    if action == "tool_call" and "tool_call" in data:
        tc = data["tool_call"]
        tool_call = ToolCallSpec(
            tool=tc.get("tool", ""),
            args=tc.get("args", {}),
            reason=tc.get("reason", ""),
        )

    memory_store = None
    if action == "store_memory" and "memory_store" in data:
        ms = data["memory_store"]
        memory_store = MemoryStoreRequest(
            key=ms.get("key", "fact"),
            value=ms.get("value", ""),
            namespace=ms.get("namespace", "global"),
        )

    return DecisionOutput(
        action=action,
        tool_call=tool_call,
        memory_store=memory_store,
        answer=data.get("answer"),
        reasoning=data.get("reasoning", ""),
        confidence=float(data.get("confidence", 0.5)),
    )
