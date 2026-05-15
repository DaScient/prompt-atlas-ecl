# Part III — Science and Discovery · Diagram

```mermaid
flowchart LR
  subgraph Theories
    QM[Quantum Mechanics]
    GR[General Relativity]
  end
  QM -. paradox .- GR
  Bridges[[AI-proposed bridges]]
  QM --> Bridges
  GR --> Bridges
  CosmicNoise([Cosmic Noise]) --> Bridges
  Bridges -->|family of models| Lattice{{Lattice of partial truths}}
  Lattice -->|interpretability gap| Ethics[/Ethics of Discovery/]
  Lattice --> Bio[Biology, Life, & Beyond]
  Bio --> NewLife[/Designed Symbiosis/]
  NewLife --> Stewardship((Stewardship))
```

*From the QM/GR paradox through AI-built bridges to a lattice of partial models, then on to designed biology and the stewardship it demands.* Anchored to [Chapter 5](../../../PROMPT_ATLAS.md#ch5-quantum-bridges-and-cosmic-noise) and [Chapter 6](../../../PROMPT_ATLAS.md#ch6-biology-life-and-beyond). Repo touch-points: [`losses_geom.py`](../../../src/losses_geom.py), [`z3_tester.py`](../../../src/testers/z3_tester.py).
