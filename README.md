# EAG Session 6 — Cognitive Agent (TRACE Architecture)

**Typed Reactive Agent with Cognitive Envelopes**

---

## Architecture

The agent implements a strict four-layer cognitive pipeline. Every layer boundary
is a typed Pydantic v2 contract — no free-form dict passing between roles.

```
┌─────────────────────────────────────────────────────────────┐
│                     agent6.py (main loop)                   │
│                                                             │
│  PerceptionInput ──► PERCEPTION ──► PerceptionOutput        │
│                            │                                │
│              (entity-guided memory search)                  │
│                            ▼                                │
│  MemoryLookupRequest ──► MEMORY ──► MemoryLookupResult      │
│                            │                                │
│  DecisionInput ────────► DECISION ──► DecisionOutput        │
│                            │                                │
│  ActionRequest ─────────► ACTION ──► ActionResult           │
│                            │                                │
│                     MCP stdio transport                     │
│                     (mcp_server.py tools)                   │
└─────────────────────────────────────────────────────────────┘
```

### Key innovations

| Feature | Description |
|---------|-------------|
| **Entity-guided memory** | Perception extracts entities first; memory lookup uses those entities as search terms — improving recall precision |
| **Confidence-bounded iterations** | Decision emits a `confidence` score; `final_answer` + high confidence terminates the loop early, bounding iteration count |
| **Cognitive routing via Gateway V3** | `auto_route="perception"` → TINY tier (fast/cheap classifier); `auto_route="decision"` → LARGE tier (capable reasoner) |
| **Persistent MCP session** | One MCP subprocess per agent run; no per-call startup overhead |
| **Durable episodic memory** | `state/memory.json` survives across runs — enables Query C's cross-run retrieval |

### Module map

| File | Role | Input → Output |
|------|------|----------------|
| `schemas.py` | Pydantic v2 contracts | (no I/O — all types live here) |
| `perception.py` | Intent extraction | `PerceptionInput` → `PerceptionOutput` |
| `memory.py` | Durable key-value store | `MemoryLookupRequest` → `MemoryLookupResult` / `MemoryStoreRequest` → `MemoryFact` |
| `decision.py` | Action planning | `DecisionInput` → `DecisionOutput` |
| `action.py` | MCP tool execution | `ActionRequest` → `ActionResult` |
| `agent6.py` | Main loop + CLI | wires all layers |
| `mcp_server.py` | MCP tool server (existing) | 9 tools over stdio |

---

## Prerequisites

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in TAVILY_API_KEY and at least one LLM provider key
```

### 3. Start LLM Gateway V3

In a separate terminal:

```bash
cd "c189e31f-c189-404a-95d8-0ba8b0756c9e/llm_gatewayV3"
uv run main.py
# Gateway starts on http://localhost:8101
```

---

## Running the four target queries

### Query A — ECB headquarters time
*Expected: 3 iterations (web_search → get_time → final_answer)*

```bash
uv run agent6.py --query-a
```

**Query text:**
> What is the current time in the city where the European Central Bank is headquartered?

---

### Query B — Multi-hop currency conversion
*Expected: 3 iterations (GBP→USD → USD→JPY → final_answer)*

```bash
uv run agent6.py --query-b
```

**Query text:**
> I have 500 British Pounds. First convert them to US Dollars, then convert that result to Japanese Yen. What is the final amount in JPY?

---

### Query C — Durable memory (two runs)

**Run 1 — store the fact:**
*Expected: 1 iteration (store_memory + acknowledge)*

```bash
uv run agent6.py --query-c1
```

**Query text:**
> Note down this fact for future reference: The best framework for building AI agents in Python is LangGraph.

**Run 2 — recall from memory:**
*Expected: 1 iteration (memory lookup → final_answer)*

```bash
uv run agent6.py --query-c2
```

**Query text:**
> What is the best framework for building AI agents in Python, based on what you know about me?

---

### Query D — Research + file operations
*Expected: 4 iterations (web_search → create_file → read_file → final_answer)*

```bash
uv run agent6.py --query-d
```

**Query text:**
> Find the three most popular Python web frameworks, save them to a file called 'python_frameworks.txt' in the sandbox, then verify the file was created correctly by reading it back and reporting the contents.

---

## Custom query

```bash
uv run agent6.py "What time is it in Tokyo right now?"
```

---

## Clean state between runs

```bash
uv run agent6.py --clean
```

This deletes `state/` (memory) and `sandbox/` (MCP file sandbox).

---

## Terminal output

*(Captured from clean state — replace with actual output before submission)*

```
Query A output here
```

```
Query B output here
```

```
Query C1 output here
Query C2 output here
```

```
Query D output here
```

---

## YouTube demo

[Link to be added — demo of all four queries end-to-end]

---

## Constraints satisfied

- [x] Pydantic v2 on every boundary (`schemas.py`)
- [x] `uv` for dependency management (`pyproject.toml`)
- [x] MCP server stdio transport (`action.py` → `mcp_server.py`)
- [x] No third-party agentic frameworks (LangGraph/LangChain/CrewAI)
- [x] LLM Gateway V3 for every LLM call (`auto_route="perception"` / `"decision"`)
- [x] Durable memory in `state/` (Query C cross-run behaviour)
- [x] `state/` cleanable between attempts (`--clean` flag)
- [x] No regex on LLM output (JSON parsed with `str.find` + `json.loads`)
