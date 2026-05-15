# Appendix C · Case Studies & Speculative Vignettes

> **Canonical:** [PROMPT_ATLAS.md#appendices](../../../PROMPT_ATLAS.md#appendices) · **Cross-links:** "Case Study" sections in chapter expansions.

The author lists five flagship case studies. Each is fully developed in its source chapter; this appendix indexes them and adds repo cross-links for builders.

| # | Case Study | Source | Expansion | Repo touch-points |
|---|------------|--------|-----------|-------------------|
| C.1 | **The Martian Charter** — A hybrid AI-human constitution forged on the first colony. | [Ch. 9](../../../PROMPT_ATLAS.md#ch9-martian-republics-and-alien-treaties) | [`ch09`](../expansions/ch09-martian-republics-and-alien-treaties.md) | [`MACP bus`](../../../src/macp/bus.py) (chamber-as-topic), [`Z3 Tester`](../../../src/testers/z3_tester.py) (concurrent-passage invariants) |
| C.2 | **The Coral Whisperer** — AI restores reefs with designed symbiosis. | [Ch. 6](../../../PROMPT_ATLAS.md#ch6-biology-life-and-beyond) | [`ch06`](../expansions/ch06-biology-life-and-beyond.md) | [`Z3 Tester`](../../../src/testers/z3_tester.py) (kill-switch invariants), [`tracking/`](../../../src/tracking/__init__.py) (tier go/no-go logging) |
| C.3 | **Festival of Echoes** — A yearly carnival where humans and AIs share memory-masks. | [Ch. 13](../../../PROMPT_ATLAS.md#ch13-carnival-of-prompts) | [`ch13`](../expansions/ch13-carnival-of-prompts.md) | [`MACP bus`](../../../src/macp/bus.py) (curator/weaver/trickster agents), [`CoLearningMemoryStore`](../../../src/vectorstore/qdrant_store.py) (festival archive) |
| C.4 | **The Pandemic Oracle** — AI cuts death toll in half through predictive simulation. | [Ch. 11](../../../PROMPT_ATLAS.md#ch11-preparing-for-collapse-and-renewal) | [`ch11`](../expansions/ch11-preparing-for-collapse-and-renewal.md) | sentinel agents on [`MACP bus`](../../../src/macp/bus.py); calibration logs via [`tracking/`](../../../src/tracking/__init__.py) |
| C.5 | **The Wormhole Ledger** — Wormhole navigation coordinates become galactic Bitcoin. | [Ch. 10](../../../PROMPT_ATLAS.md#ch10-information-as-cosmic-currency) | [`ch10`](../expansions/ch10-information-as-cosmic-currency.md) | quorum patterns on [`MACP bus`](../../../src/macp/bus.py); sharded archives on [`CoLearningMemoryStore`](../../../src/vectorstore/qdrant_store.py) |

## Guide for AI & Humanity

- **Vignettes are rehearsals, not blueprints.** Treat each case study as a scenario to *practice* with — pre-mortem, dry-run, after-action — never as a turn-key product spec.
- **Always restore the safeguards.** Each case study compresses ethical scaffolding for narrative readability; the corresponding chapter expansion restores the full reversibility, consent, audit, and human-in-the-loop requirements.
- **Cite the source.** When using a vignette in a talk, paper, or workshop, link the chapter and credit the author.
