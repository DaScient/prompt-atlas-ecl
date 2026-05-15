# Prompt Atlas — Research Notes

This directory contains research-grade documentation for the Prompt
Atlas Entanglement Co-Learning engine. The goal is to give readers
enough to reproduce results and to cite the project precisely.

## Reproducible benchmarks

The benchmark harness lives at `scripts/bench.py` and writes a JSON
document whose schema is locked at
[`bench_result.schema.json`](./bench_result.schema.json). The schema
is versioned (`version: 1`) so downstream notebooks can detect a
breaking change.

### Quickstart

```bash
# Deterministic torch path — bit-for-bit reproducible.
python -m scripts.bench --steps 16 --json /tmp/bench-torch.json

# Phase 5 orchestrator path with a prompt-pack template.
python -m scripts.bench \
    --llm \
    --pack myth-1 --prompt user_story \
    --var system_under_review="a feed-ranking service" \
    --steps 16 \
    --json /tmp/bench-myth.json
```

### Headline metrics

| Metric | Meaning |
|---|---|
| `e_star_mean` | average E★ across all steps; primary coherence number |
| `e_star_final` | E★ at the final step; how converged the run was |
| `drift_mean` | average L2 between consecutive latent states |
| `steps_per_second` | throughput; useful for regression-tracking |

## Reading list

The framing in this repo borrows from a handful of canonical lines of
work:

* **Information bottleneck / InfoNCE** — Oord et al. 2018, "Representation Learning with Contrastive Predictive Coding".
* **Optimal transport for representations** — Cuturi 2013, "Sinkhorn Distances".
* **Maximum Mean Discrepancy** — Gretton et al. 2012.
* **Multi-agent debate / verification** — Du et al. 2023, "Improving Factuality and Reasoning in Language Models through Multiagent Debate".

## Citing the project

A machine-readable `CITATION.cff` lives at the repo root. Update it
when shipping a tagged release; bench results in published work should
reference the exact commit SHA.
