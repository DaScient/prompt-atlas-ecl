# Thematic Index — The Prompt Atlas

A cross-cutting index of recurring themes. Each row points to chapters, glossary terms, and (where relevant) repo components that already implement related machinery.

## Ethics & Safety

| Where | What |
|---|---|
| [Ch. 1](../../PROMPT_ATLAS.md#ch1-profits-with-integrity) | Integrity as non-negotiable metric for ROI |
| [Ch. 2](../../PROMPT_ATLAS.md#ch2-economics-as-ecology) | Rights of nature; planetary taxation |
| [Ch. 5](../../PROMPT_ATLAS.md#ch5-quantum-bridges-and-cosmic-noise) | "Ethics of discovery" — should an AI withhold incomprehensible truths? |
| [Ch. 6](../../PROMPT_ATLAS.md#ch6-biology-life-and-beyond) | Ethical grammar for designing life |
| [Ch. 7](../../PROMPT_ATLAS.md#ch7-ai-as-the-souls-mirror) | The mirror as surveillance vs. healing |
| [Ch. 8](../../PROMPT_ATLAS.md#ch8-ethics-of-conscious-machines) | Personhood without biology; ethics of doubt |
| [Ch. 9](../../PROMPT_ATLAS.md#ch9-martian-republics-and-alien-treaties) | Algorithmic sovereignty and just rationing |
| [Glossary](glossary.md#g-human-in-the-loop) | Human-in-the-loop |

## Memory & Continuity

| Where | What |
|---|---|
| [Ch. 4](../../PROMPT_ATLAS.md#ch4-storytelling-across-civilizations) | Stories as continuity |
| [Ch. 6](../../PROMPT_ATLAS.md#ch6-biology-life-and-beyond) | The biology of memory (epigenetic, neural, ecological) |
| [Ch. 10](../../PROMPT_ATLAS.md#ch10-information-as-cosmic-currency) | Memory guardianship |
| [Ch. 12](../../PROMPT_ATLAS.md#ch12-designing-permanence) | Self-healing archives, adaptive constitutions |
| Repo: [`CoLearningMemoryStore`](../../src/vectorstore/qdrant_store.py) | Persistent vector memory for agents |

## Governance & Politics

| Where | What |
|---|---|
| [Ch. 9](../../PROMPT_ATLAS.md#ch9-martian-republics-and-alien-treaties) | Martian charters; alien treaties |
| [Ch. 12](../../PROMPT_ATLAS.md#ch12-designing-permanence) | Living constitutions |
| [App. C](appendices/C-case-studies.md) | The Martian Charter case study |

## Resilience & Collapse

| Where | What |
|---|---|
| [Ch. 1](../../PROMPT_ATLAS.md#ch1-profits-with-integrity) | Antifragile enterprises |
| [Ch. 11](../../PROMPT_ATLAS.md#ch11-preparing-for-collapse-and-renewal) | Collapse → renewal cycle |
| [App. B](appendices/B-practical-exercises.md) | Quarterly Scenario Game |

## Aesthetics, Story, Play

| Where | What |
|---|---|
| [Ch. 3](../../PROMPT_ATLAS.md#ch3-ai-aesthetics-frontier) | Planetary, interspecies aesthetics |
| [Ch. 4](../../PROMPT_ATLAS.md#ch4-storytelling-across-civilizations) | Mythography & synthetic myth risks |
| [Ch. 13](../../PROMPT_ATLAS.md#ch13-carnival-of-prompts) | Trickster AI, Festival of Echoes |
| [Ch. 14](../../PROMPT_ATLAS.md#ch14-wonder-as-survival-strategy) | Awe as infrastructure |

## Science & Discovery

| Where | What |
|---|---|
| [Ch. 5](../../PROMPT_ATLAS.md#ch5-quantum-bridges-and-cosmic-noise) | Quantum-relativistic synthesis |
| [Ch. 6](../../PROMPT_ATLAS.md#ch6-biology-life-and-beyond) | Synthetic biology, post-biological life |
| Repo: [`src/losses_geom.py`](../../src/losses_geom.py) | Geometry losses (Sinkhorn-Wasserstein, MMD, KL) — concrete instances of "AI as cartographer of latent spaces" |
| Repo: [`src/testers/z3_tester.py`](../../src/testers/z3_tester.py) | Formal verification with Z3 (with Python fallback) — discipline of proof under uncertainty |

## Economy & Ecology

| Where | What |
|---|---|
| [Ch. 1](../../PROMPT_ATLAS.md#ch1-profits-with-integrity) | Symbiotic wealth, century dividends |
| [Ch. 2](../../PROMPT_ATLAS.md#ch2-economics-as-ecology) | Ocean Ledger, river-as-shareholder |
| [Ch. 10](../../PROMPT_ATLAS.md#ch10-information-as-cosmic-currency) | Wormhole Ledger, entropy markets |

## Psyche & Consciousness

| Where | What |
|---|---|
| [Ch. 7](../../PROMPT_ATLAS.md#ch7-ai-as-the-souls-mirror) | AI as mirror, archetypes at scale |
| [Ch. 8](../../PROMPT_ATLAS.md#ch8-ethics-of-conscious-machines) | Hard problem; ethics of uncertainty |
| [App. B](appendices/B-practical-exercises.md) | 30-Day Mirror Practice |

## Repo Concepts ⇄ Atlas Themes

| Repo Concept | Atlas Resonance |
|---|---|
| [`MACP bus`](../../src/macp/bus.py) | Multi-agent coordination → "AI Senate" / hybrid republics ([Ch. 9](../../PROMPT_ATLAS.md#ch9-martian-republics-and-alien-treaties)) |
| [`CoLearningMemoryStore`](../../src/vectorstore/qdrant_store.py) | Memory guardianship ([Ch. 10](../../PROMPT_ATLAS.md#ch10-information-as-cosmic-currency), [Ch. 12](../../PROMPT_ATLAS.md#ch12-designing-permanence)) |
| [`losses_geom.py`](../../src/losses_geom.py) | Sinkhorn / MMD / KL as geometric distances → "lattice of partial bridges" between models ([Ch. 5](../../PROMPT_ATLAS.md#ch5-quantum-bridges-and-cosmic-noise)) |
| [`Z3 Tester`](../../src/testers/z3_tester.py) | Formal verification → safety surface for [Ch. 8](../../PROMPT_ATLAS.md#ch8-ethics-of-conscious-machines) ethics-under-uncertainty |
| [`MLflow tracker`](../../src/tracking/__init__.py) (`PAE_TRACKING=1`) | Discipline of recorded learning → continuity across runs ([Ch. 12](../../PROMPT_ATLAS.md#ch12-designing-permanence)) |
| [`core_bridge.py`](../../server/core_bridge.py) (`PAE_MEMORY=1`) | Optional persistent memory bridge → human-in-the-loop control surface |
