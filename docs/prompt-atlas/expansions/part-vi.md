# Part VI — Resilience and Survival · Expansion

> **Diagram:** [`diagrams/part-vi-resilience.md`](../diagrams/part-vi-resilience.md) · **Chapters:** [Ch. 11](../../../PROMPT_ATLAS.md#ch11-preparing-for-collapse-and-renewal), [Ch. 12](../../../PROMPT_ATLAS.md#ch12-designing-permanence)

## Why this Part exists

Part VI accepts that collapse will visit and asks how civilizations *bend without breaking*. Ch. 11 builds the sentinel/absorption/renewal loop; Ch. 12 redefines permanence as *resilient transformation* — not stasis.

## Through-lines

- **Foresight without listening is theatre.** Sentinels matter only if their warnings are heeded.
- **Renewal is a designable practice**, not an accident.
- **Permanence is a river, not a wall.**

## Repo touch-points

| Concept | Repo |
|---|---|
| Sentinel agents | [`MACP bus`](../../../src/macp/bus.py) |
| Self-healing archive | [`CoLearningMemoryStore`](../../../src/vectorstore/qdrant_store.py) |
| Constraints / invariants | [`testers/z3_tester.py`](../../../src/testers/z3_tester.py) |
| Tracking long-running runs | [`tracking/__init__.py`](../../../src/tracking/__init__.py) — `PAE_TRACKING=1` |

## Guide for AI & Humanity

- **Plan to fail gracefully.** Optimize for the worst quartile, not the median.
- **Permanence is renewal.** Build for migration, not for stone.
- **Listen-or-lose.** The sentinel that no one heeds is wasted infrastructure.
