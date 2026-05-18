"""
Pydantic v2 contracts for all agent cognitive-layer boundaries.
No free-form dict passing between layers — every handoff is typed here.
"""
from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ── Perception layer ──────────────────────────────────────────────────────────

class PerceptionInput(BaseModel):
    query: str
    iteration: int
    history: list[dict[str, Any]]


class PerceptionOutput(BaseModel):
    intent: str
    entities: list[str]
    required_actions: list[str]
    is_answerable: bool
    confidence: float = Field(ge=0.0, le=1.0)
    focus: str


# ── Memory layer ──────────────────────────────────────────────────────────────

class MemoryStoreRequest(BaseModel):
    key: str
    value: str
    namespace: str = "global"


class MemoryLookupRequest(BaseModel):
    query: str
    namespace: str = "global"
    max_results: int = 5


class MemoryFact(BaseModel):
    key: str
    value: str
    namespace: str
    created_at: str


class MemoryLookupResult(BaseModel):
    facts: list[MemoryFact]
    hit_count: int


# ── Decision layer ────────────────────────────────────────────────────────────

class DecisionInput(BaseModel):
    goal: str
    iteration: int
    max_iterations: int
    perception: PerceptionOutput
    memory: MemoryLookupResult
    tool_results: list[dict[str, Any]]
    available_tools: list[str]


class ToolCallSpec(BaseModel):
    tool: str
    args: dict[str, Any]
    reason: str


class DecisionOutput(BaseModel):
    action: Literal["tool_call", "store_memory", "final_answer"]
    tool_call: Optional[ToolCallSpec] = None
    memory_store: Optional[MemoryStoreRequest] = None
    answer: Optional[str] = None
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


# ── Action layer ──────────────────────────────────────────────────────────────

class ActionRequest(BaseModel):
    tool: str
    args: dict[str, Any]


class ActionResult(BaseModel):
    success: bool
    tool: str
    args: dict[str, Any]
    output: Any
    error: Optional[str] = None
    duration_ms: float


# ── Agent state (cross-iteration) ─────────────────────────────────────────────

class AgentState(BaseModel):
    query: str
    iteration: int = 0
    max_iterations: int = 10
    history: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    final_answer: Optional[str] = None
    status: Literal["running", "done", "failed"] = "running"
