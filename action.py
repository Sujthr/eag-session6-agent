"""
Action layer — executes tools via MCP stdio transport.

Manages a single persistent MCP subprocess for the agent run lifetime,
eliminating per-call startup overhead. Uses the official MCP Python client
library (mcp[cli]) with the stdio transport — no tool dispatch reimplemented.

Input  → ActionRequest  (typed)
Output → ActionResult   (typed)
"""
from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from schemas import ActionRequest, ActionResult

_MCP_SCRIPT = str(Path(__file__).parent / "mcp_server.py")

_stack: contextlib.AsyncExitStack | None = None
_session: ClientSession | None = None


async def start_session() -> None:
    global _stack, _session

    _stack = contextlib.AsyncExitStack()
    await _stack.__aenter__()

    params = StdioServerParameters(
        command="uv",
        args=["run", "--no-sync", _MCP_SCRIPT],
        env=None,
    )
    read, write = await _stack.enter_async_context(stdio_client(params))
    _session = await _stack.enter_async_context(ClientSession(read, write))
    await _session.initialize()


async def stop_session() -> None:
    global _stack, _session
    if _stack is not None:
        await _stack.aclose()
    _stack = None
    _session = None


async def execute(req: ActionRequest) -> ActionResult:
    if _session is None:
        await start_session()

    t0 = time.perf_counter()
    try:
        result = await _session.call_tool(req.tool, req.args)
        elapsed = (time.perf_counter() - t0) * 1000

        output: Any = None
        if result.content:
            raw_text = result.content[0].text
            try:
                output = json.loads(raw_text)
            except (json.JSONDecodeError, AttributeError):
                output = raw_text

        return ActionResult(
            success=True,
            tool=req.tool,
            args=req.args,
            output=output,
            duration_ms=round(elapsed, 1),
        )

    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - t0) * 1000
        return ActionResult(
            success=False,
            tool=req.tool,
            args=req.args,
            output=None,
            error=str(exc),
            duration_ms=round(elapsed, 1),
        )
