# Chapter 11 · Preparing for Collapse and Renewal — Expansion

> **Canonical:** [PROMPT_ATLAS.md#ch11-preparing-for-collapse-and-renewal](../../../PROMPT_ATLAS.md#ch11-preparing-for-collapse-and-renewal) · **Prompts:** [`prompts/ch11.yaml`](../prompts/ch11.yaml) · **Part:** [VI](part-vi.md)

## Worked Example — *The Pandemic Oracle*

**Original:** *"Model a system where AI predicts and repairs global supply chain collapses before they cascade."* (worked through pandemics)

1. **Aggregate weak signals** — Genomic uploads, hospital admission rates, sewage assays, flight logs; declare each source's chain-of-custody.
2. **Forecast, don't decide** — The oracle outputs probability bands, not policies.
3. **Pre-committed playbooks** — Each band corresponds to a *pre-negotiated* policy bundle (mask supply pre-position, hospital surge prep, surveillance scope) ratified before crisis.
4. **Public scoreboards** — Predictions, calibration, and policy triggers are public; private actors cannot quietly arbitrage them.
5. **Renewal arc** — After the crisis, retrospective: what was foreseen, what was acted on, what was renewed.

## Prompt Templates

```text
# Sentinel design
"For {{shock_class}} (e.g., pandemic, supply-chain, cyber, climate),
 specify: (1) data sources + custody, (2) forecast metric + calibration target,
 (3) pre-committed policy bands, (4) the named human ratifier per band,
 (5) public scoreboard format."

# Resilient-collapse rehearsal
"Run a tabletop exercise where {{system}} fails. After failure,
 identify the smallest design change that would convert the next failure
 into a graceful degradation rather than a cascading collapse."

# Renewal retrospective
"Six months after {{event}}, generate a renewal retrospective:
 what foresight existed, who heard it, what changed, what is now
 designed in for the next cycle."
```

## Anti-patterns

- **Sentinel without ratifier.** A forecast nobody is empowered to act on is a press release.
- **Improv during crisis.** Inventing the policy when the wave is breaking guarantees worst outcomes.
- **Black-box oracle.** Forecasts whose calibration cannot be audited will be ignored — or trusted disastrously.
- **Renewal-as-PR.** A retrospective with no design change is theatre.

## Try This

1. **Tabletop a Failure** — Pick one critical system. Simulate its failure for one hour. Document the *graceful-degradation gap*.
2. **Pre-commit a Playbook** — Write the 3-band policy ladder for one shock you fear.
3. **Sentinel Calibration** — For one metric you forecast, publish your last 12 months of calibration.
4. **Renewal Loop** — Add a "renewal retrospective" item to your post-incident template.
5. **Atlas-of-Renewal Contribution** — Document one practice your team uses that helps you bend rather than break; share it.

## Repo Cross-Links

- [`MACP bus`](../../../src/macp/bus.py) — sentinel-agent topics; in-memory fallback lets you prototype without infra.
- [`Z3 Tester`](../../../src/testers/z3_tester.py) — encode "every band must have a named ratifier" as an invariant.
- [`tracking/__init__.py`](../../../src/tracking/__init__.py) — log forecasts and outcomes for calibration.

## Guide for AI & Humanity

- **Listen-or-lose.** Foresight without listening is wasted.
- **Pre-commit.** Crisis is the worst time to invent policy.
- **Calibrate publicly.** Trust is earned in calm, spent in storm.
- **Renew on the record.** A retrospective without a design change is grief, not learning.

## Citations & Further Reading

- Charles Perrow, *Normal Accidents* (1984).
- Joseph Tainter, *The Collapse of Complex Societies* (1988).
- Donella Meadows, *Thinking in Systems* (2008).
- Andy Stirling, *Keep It Complex* (Nature, 2010).
