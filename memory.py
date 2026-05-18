"""
Memory layer — durable episodic store backed by state/memory.json.
Persists across agent runs; enables Query C's cross-run retrieval.

Two retrieval modes:
  • Keyword scoring  — overlap between query words and key+value text
  • Wildcard ("*")   — return all facts in a namespace
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from schemas import (
    MemoryFact,
    MemoryLookupRequest,
    MemoryLookupResult,
    MemoryStoreRequest,
)

STATE_DIR = Path(__file__).parent / "state"
MEMORY_FILE = STATE_DIR / "memory.json"


# ── internal helpers ──────────────────────────────────────────────────────────

def _load() -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"namespaces": {}}


def _save(data: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── public API (typed contracts) ──────────────────────────────────────────────

def store(req: MemoryStoreRequest) -> MemoryFact:
    data = _load()
    ns = data["namespaces"].setdefault(req.namespace, {})
    fact = MemoryFact(
        key=req.key,
        value=req.value,
        namespace=req.namespace,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    ns[req.key] = fact.model_dump()
    _save(data)
    return fact


def lookup(req: MemoryLookupRequest) -> MemoryLookupResult:
    data = _load()
    ns = data["namespaces"].get(req.namespace, {})

    if not ns:
        return MemoryLookupResult(facts=[], hit_count=0)

    if req.query.strip() == "*":
        facts = [MemoryFact(**v) for v in list(ns.values())[: req.max_results]]
        return MemoryLookupResult(facts=facts, hit_count=len(ns))

    query_words = set(req.query.lower().split())
    scored: list[tuple[int, MemoryFact]] = []
    for record in ns.values():
        content = (record["key"] + " " + record["value"]).lower()
        score = sum(1 for w in query_words if w in content)
        if score > 0:
            scored.append((score, MemoryFact(**record)))

    scored.sort(key=lambda x: x[0], reverse=True)
    facts = [f for _, f in scored[: req.max_results]]
    return MemoryLookupResult(facts=facts, hit_count=len(scored))


def get_all(namespace: str = "global") -> list[MemoryFact]:
    data = _load()
    return [MemoryFact(**v) for v in data["namespaces"].get(namespace, {}).values()]


def clear(namespace: str | None = None) -> int:
    data = _load()
    if namespace is None:
        count = sum(len(ns) for ns in data["namespaces"].values())
        data["namespaces"] = {}
    else:
        ns = data["namespaces"].get(namespace, {})
        count = len(ns)
        data["namespaces"][namespace] = {}
    _save(data)
    return count
