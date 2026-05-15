# Part V — Intergalactic Horizons · Expansion

> **Diagram:** [`diagrams/part-v-intergalactic.md`](../diagrams/part-v-intergalactic.md) · **Chapters:** [Ch. 9](../../../PROMPT_ATLAS.md#ch9-martian-republics-and-alien-treaties), [Ch. 10](../../../PROMPT_ATLAS.md#ch10-information-as-cosmic-currency)

## Why this Part exists

Part V projects the ethical and political work outward — to colonies, to first contact, to economies of pure information. Ch. 9 introduces *algorithmic sovereignty* and *temporal sovereignty*; Ch. 10 reframes wealth itself as compressed knowledge.

## Through-lines

- **Survival is shared infrastructure** — every breath in a Martian dome is a coordination problem.
- **Information beats matter at scale** — across light-years, knowledge is the rare coin.
- **Memory guardianship as duty** — preserving truth against entropy and manipulation is itself political work.

## Repo touch-points

| Concept | Repo |
|---|---|
| AI Senate / hybrid governance | [`MACP bus`](../../../src/macp/bus.py) — agent topics with quorum patterns |
| Memory guardianship | [`CoLearningMemoryStore`](../../../src/vectorstore/qdrant_store.py) |
| Latency-tolerant coordination | bus topics with retry / fallback (see `src/macp/bus.py` lazy import + in-memory fallback) |

## Guide for AI & Humanity

- **No oxygen ownership.** Universal access to survival commons is non-negotiable.
- **Algorithmic sovereignty needs human appeal.** Always.
- **Treaties precede contact.** Draft your protocols before you need them.
