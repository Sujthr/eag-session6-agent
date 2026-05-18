"""
Perception layer — extracts structured intent from raw query + history.

Routes through LLM Gateway V3 with auto_route="perception" so the gateway
picks a fast/cheap TINY-tier model for this classification task.

Input  → PerceptionInput  (typed)
Output → PerceptionOutput (typed, Pydantic-validated)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Resolve gateway client without installing as a package
sys.path.insert(
    0,
    str(
        Path(__file__).parent
        / "c189e31f-c189-404a-95d8-0ba8b0756c9e"
        / "llm_gatewayV3"
    ),
)
from client import LLM  # noqa: E402  (path-inserted above)

from schemas import PerceptionInput, PerceptionOutput

_llm = LLM()

_SYSTEM = """\
You are the Perception module of a multi-layer AI agent.
Analyse the current query and conversation history, then output a JSON object
that tells the rest of the agent what the user wants and what to do next.

Output ONLY valid JSON — no markdown fences, no prose.

Available tools the agent can call:
  web_search, fetch_url, get_time, currency_convert,
  read_file, list_dir, create_file, update_file, edit_file, store_memory

JSON fields:
  intent         — one clear sentence: what the user ultimately wants
  entities       — list of key named things (places, currencies, filenames, topics)
  required_actions — ordered list of tools / steps still needed to answer
  is_answerable  — true only if the history already contains everything needed
  confidence     — float 0.0–1.0: how certain we can answer RIGHT NOW without more tools
  focus          — one action the agent should take THIS iteration

Rules:
  • If history shows the last tool returned an error, focus on recovering or trying a different approach.
  • If is_answerable is true, set confidence >= 0.85 and focus = "give final answer".
  • Keep required_actions short (≤ 4 items)."""


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start <= end:
        text = text[start : end + 1]
    return json.loads(text)


def perceive(inp: PerceptionInput) -> PerceptionOutput:
    history_str = ""
    if inp.history:
        recent = inp.history[-6:]
        history_str = "\n\nConversation so far:\n" + "\n".join(
            "[{role}] {content}".format(
                role=h.get("role", "?"),
                content=str(h.get("content", ""))[:300],
            )
            for h in recent
        )

    prompt = (
        f"Query: {inp.query}\n"
        f"Iteration: {inp.iteration}"
        f"{history_str}\n\n"
        "Output JSON:"
    )

    resp = _llm.chat(
        prompt=prompt,
        system=_SYSTEM,
        auto_route="perception",
        response_format={"type": "json_object"},
        max_tokens=512,
        temperature=0.1,
    )

    raw = resp.get("text", "")
    parsed = resp.get("parsed") or _parse_json(raw)
    return PerceptionOutput.model_validate(parsed)
