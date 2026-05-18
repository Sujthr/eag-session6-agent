"""
EAG Session 6 — Cognitive Agent (TRACE architecture)
Typed Reactive Agent with Cognitive Envelopes

Pipeline per iteration:
    Perception → (entity-guided) Memory → Decision → Action

All layer boundaries use Pydantic v2 typed contracts (schemas.py).
LLM calls route through LLM Gateway V3 (auto_route per cognitive role).
Tools execute via MCP stdio transport (action.py).
Memory persists across runs in state/memory.json (memory.py).

Usage:
    uv run agent6.py "your query here"
    uv run agent6.py --query-a       # ECB headquarters time
    uv run agent6.py --query-b       # GBP → USD → JPY multi-hop currency
    uv run agent6.py --query-c1      # Store memory (run first)
    uv run agent6.py --query-c2      # Recall memory (run second)
    uv run agent6.py --query-d       # Research + file ops
    uv run agent6.py --clean         # Wipe state/ and sandbox/
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import action
import decision
import memory
import perception
from schemas import (
    ActionRequest,
    AgentState,
    DecisionInput,
    MemoryLookupRequest,
    PerceptionInput,
)

# ── Four target queries ───────────────────────────────────────────────────────

TARGET_QUERIES: dict[str, str] = {
    "a": (
        "What is the current time in the city where the European Central Bank "
        "is headquartered?"
    ),
    "b": (
        "I have 500 British Pounds. First convert them to US Dollars, then "
        "convert that result to Japanese Yen. What is the final amount in JPY?"
    ),
    "c1": (
        "Note down this fact for future reference: "
        "The best framework for building AI agents in Python is LangGraph."
    ),
    "c2": (
        "What is the best framework for building AI agents in Python, "
        "based on what you know about me?"
    ),
    "d": (
        "Find the three most popular Python web frameworks, save them to a file "
        "called 'python_frameworks.txt' in the sandbox, then verify the file was "
        "created correctly by reading it back and reporting the contents."
    ),
}

AVAILABLE_TOOLS = [
    "web_search", "fetch_url", "get_time", "currency_convert",
    "read_file", "list_dir", "create_file", "update_file", "edit_file",
]

_DIVIDER = "─" * 60


# ── Clean helper ──────────────────────────────────────────────────────────────

def _clean() -> None:
    base = Path(__file__).parent
    for name in ("state", "sandbox"):
        p = base / name
        if p.exists():
            shutil.rmtree(p)
            print(f"Removed {p}")
        else:
            print(f"Nothing to remove at {p}")
    print("Clean complete.")


# ── Agent loop ────────────────────────────────────────────────────────────────

async def run_agent(query: str, max_iterations: int = 8) -> AgentState:
    state = AgentState(query=query, max_iterations=max_iterations)

    print(f"\n{_DIVIDER}")
    print(f"QUERY : {query}")
    print(_DIVIDER)

    await action.start_session()

    try:
        while state.status == "running" and state.iteration < state.max_iterations:
            state.iteration += 1
            print(f"\n[Iteration {state.iteration}/{state.max_iterations}]")

            # ── 1. PERCEPTION ─────────────────────────────────────────
            perc_in = PerceptionInput(
                query=query,
                iteration=state.iteration,
                history=state.history[-6:],
            )
            perc_out = perception.perceive(perc_in)
            print(f"  Perception | intent : {perc_out.intent}")
            print(f"             | focus  : {perc_out.focus}")
            print(f"             | entities: {perc_out.entities}")

            # ── 2. MEMORY (entity-guided lookup) ──────────────────────
            search_terms = " ".join(perc_out.entities) if perc_out.entities else query
            mem_result = memory.lookup(
                MemoryLookupRequest(query=search_terms, namespace="global")
            )
            if mem_result.facts:
                print(f"  Memory     | {len(mem_result.facts)} fact(s) recalled")
                for f in mem_result.facts:
                    print(f"             | {f.key}: {f.value}")

            # ── 3. DECISION ───────────────────────────────────────────
            dec_in = DecisionInput(
                goal=query,
                iteration=state.iteration,
                max_iterations=state.max_iterations,
                perception=perc_out,
                memory=mem_result,
                tool_results=state.tool_results[-5:],
                available_tools=AVAILABLE_TOOLS,
            )
            dec_out = decision.decide(dec_in)
            print(f"  Decision   | action : {dec_out.action}  conf={dec_out.confidence:.2f}")
            print(f"             | reason : {dec_out.reasoning}")

            # ── 4. ACT ────────────────────────────────────────────────
            if dec_out.action == "final_answer":
                state.final_answer = dec_out.answer
                state.status = "done"

            elif dec_out.action == "store_memory" and dec_out.memory_store:
                fact = memory.store(dec_out.memory_store)
                print(f"  Memory     | stored  {fact.key!r} = {fact.value!r}")
                state.history.append({
                    "role": "memory",
                    "content": f"Stored '{fact.key}': {fact.value}",
                })
                # Decision includes acknowledgment answer when storing memory
                if dec_out.answer:
                    state.final_answer = dec_out.answer
                    state.status = "done"

            elif dec_out.action == "tool_call" and dec_out.tool_call:
                tc = dec_out.tool_call
                print(f"  Action     | {tc.tool}({json.dumps(tc.args)})")
                act_res = await action.execute(ActionRequest(tool=tc.tool, args=tc.args))

                if act_res.success:
                    snippet = json.dumps(act_res.output)[:300]
                    print(f"  Result     | {snippet}")
                else:
                    print(f"  Error      | {act_res.error}")

                state.tool_results.append(act_res.model_dump())
                state.history.append({
                    "role": "tool",
                    "tool": tc.tool,
                    "args": tc.args,
                    "content": (
                        json.dumps(act_res.output)[:600]
                        if act_res.success
                        else f"ERROR: {act_res.error}"
                    ),
                })

            else:
                state.final_answer = "Agent could not determine a valid action."
                state.status = "failed"

    finally:
        await action.stop_session()

    if state.status == "running":
        state.status = "failed"
        state.final_answer = "Max iterations reached without a final answer."

    print(f"\n{_DIVIDER}")
    print(f"Status: {state.status}  |  Iterations used: {state.iteration}")
    print(_DIVIDER)
    if state.final_answer:
        print(f"\n{state.final_answer}\n")

    return state


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        sys.exit(1)

    if args[0] == "--clean":
        _clean()
        sys.exit(0)

    alias_map = {f"--query-{k}": v for k, v in TARGET_QUERIES.items()}
    query = alias_map.get(args[0]) or " ".join(args)

    state = asyncio.run(run_agent(query))
    sys.exit(0 if state.status == "done" else 1)


if __name__ == "__main__":
    main()
