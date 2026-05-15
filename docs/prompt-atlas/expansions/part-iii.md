# Part III — Science and Discovery · Expansion

> **Diagram:** [`diagrams/part-iii-science.md`](../diagrams/part-iii-science.md) · **Chapters:** [Ch. 5](../../../PROMPT_ATLAS.md#ch5-quantum-bridges-and-cosmic-noise), [Ch. 6](../../../PROMPT_ATLAS.md#ch6-biology-life-and-beyond)

## Why this Part exists

Part III takes AI from being our calculator to being our **co-explorer of the impossible**. Ch. 5 grapples with the QM/GR chasm and asks whether AI will help us build *a lattice of partial bridges* — even ones we cannot fully comprehend. Ch. 6 turns inward to biology: AI as not just reader of the library of life but co-author of new pages.

## Through-lines

- **Interpretability vs. accuracy** — both chapters surface the same dilemma: do we accept models that predict but cannot be narrated?
- **Co-authorship with the more-than-human** — particles, fields, microbes, ecosystems all become collaborators.
- **Stewardship over conquest** — the moral center is humility before what we still don't understand.

## Repo touch-points

| Concept | Repo |
|---|---|
| Geometric distances between models | [`losses_geom.py`](../../../src/losses_geom.py) — Sinkhorn-Wasserstein, MMD, Gaussian KL |
| Formal verification under uncertainty | [`testers/z3_tester.py`](../../../src/testers/z3_tester.py) (with Python fallback) |
| Tracking exploratory runs | [`tracking/__init__.py`](../../../src/tracking/__init__.py) (`PAE_TRACKING=1`) |
| ECL co-thinking | [`train_ecl.py`](../../../src/train_ecl.py), [`state_bus.py`](../../../src/state_bus.py) |

## Guide for AI & Humanity

- **No oracle without interpretability budget.** A model whose predictions cannot be questioned is a tyrant in lab clothes. Always allocate explicit interpretability work.
- **Reproducibility is dignity.** Track everything; the alternative is folklore.
- **Bio-design needs reversibility.** Anything released into a shared biosphere must have an off switch.
