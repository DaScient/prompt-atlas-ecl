# Chapter 1 · Profits with Integrity — Expansion

> **Canonical prose:** [PROMPT_ATLAS.md#ch1-profits-with-integrity](../../../PROMPT_ATLAS.md#ch1-profits-with-integrity)
> **Prompts (YAML):** [`prompts/ch01.yaml`](../prompts/ch01.yaml) · **Part:** [I — Prosperity and Purpose](part-i.md)

## Worked Example — *Century Dividends*

**Original prompt** (from chapter): *"Design a financial instrument that pays profits not just to current investors but to descendants one hundred years from now."*

**End-to-end walk-through:**

1. **Frame the instrument's term** — A 100-year covenant bond with two yield streams: an annual coupon to current holders and a *deferred yield* paid into a custodial trust whose beneficiaries are determined at year 100.
2. **Define the trust's beneficiary rule** — Beneficiaries can be (a) any human born in the issuing region during years 80–100, (b) restoration of an ecological asset specified at issuance (a watershed, a coral reef), or (c) a chartered institution with a public mandate. The choice is locked at issuance to prevent capture.
3. **Solve indexation** — Coupons indexed to *Gross Planetary Well-Being* (see [glossary](../glossary.md#g-gross-planetary-well-being)) rather than CPI alone, so that an issuer who externalizes harm pays more.
4. **Resolve enforcement** — Custodial trust governance is split: human trustees + an AI auditor with read-only access to the issuer's planetary metrics, escalating to humans on anomaly.
5. **Stress-test** — What happens if the issuer dissolves at year 40? (Insurance pool.) What if the beneficiary class doesn't exist at year 100? (Cascading rules to ecological asset restoration.)

## Prompt Templates

```text
# Antifragile diagnosis
"Treating {{my_organization}} as a {{biological_analog: rainforest|immune_system|coral_reef}},
 identify the three feedback loops that *strengthen* it under disruption,
 the three that weaken it, and one ritual we can adopt this quarter
 to flip a weakening loop into a strengthening one."

# Symbiotic-wealth balance sheet
"Restate {{company}}'s last annual report as a four-column balance sheet:
 (1) Financial capital, (2) Carbon balance, (3) Biodiversity contribution,
 (4) Community trust. For each row, mark which AI signal would justify the figure
 and where a human auditor must verify."

# Ethical arbitrage screen
"Given a list of {{market_inefficiencies}}, return only those whose exploitation
 also produces a measurable public good (specify which good, who benefits,
 how it is measured)."
```

## Anti-patterns

- **Composite-index myopia.** Collapsing carbon + biodiversity + equity into a single number invites Goodhart's Law. Keep the vector.
- **"AI says it's regenerative."** A model's output is not an audit. Always require an external attestation chain (sensor data, third-party verifier, or smart-contract receipt).
- **Greenwashing prompts.** *"Justify why our product is sustainable"* trains a model to rationalize. Prefer *"Identify the three strongest reasons our product is **not** sustainable, ranked by evidence."*
- **Time-horizon laundering.** A 30-year promise paid in inflated currency is a 1-year promise in disguise. Index to physical or ecological assets.

## Try This (5 exercises)

1. **Quarterly Symbiosis Review** — Once per quarter, run the *Symbiotic-wealth balance sheet* prompt against your team or business unit. Archive results — the *delta* across quarters is the actual signal.
2. **Antifragile Diagnosis** — Pick one biological analog. Map your org. Adopt one ritual change for 90 days.
3. **Carbon Inversion Sketch** — In one paragraph, describe a transaction your organization currently profits from where a *carbon-negative* re-design would be more profitable in a 10-year horizon.
4. **Royalty Economy Thought Experiment** — If users earned a dividend whenever their data trained one of your models, what would change about your product roadmap?
5. **Bond a Future** — Write the term sheet of a *century dividend* bond your team would actually issue.

## Repo Cross-Links

- [`Z3 Tester`](../../../src/testers/z3_tester.py) — formal "integrity auditor" pattern with Python fallback (no z3 install required).
- [`losses_geom.py`](../../../src/losses_geom.py) — Sinkhorn-Wasserstein, MMD, and KL distances are useful when you need a *vector* distance between distributions instead of a scalar score (avoids composite-index myopia above).
- [`MACP bus`](../../../src/macp/bus.py) — pattern for agents (e.g., a "regeneration auditor" agent and a "human ratifier" agent) to coordinate without coupling.
- See `infra/docker-compose.yml` for optional NATS / Qdrant / MLflow.

## Guide for AI & Humanity

- **Integrity is a vector, not a scalar.** Resist the temptation to let AI emit a single "integrity score". Carry the components separately.
- **Audit trails are dignity.** Anyone whose data, land, or labor is summarized into a number deserves to see (and contest) the chain that produced it.
- **Slow signals matter.** Ecological dividends accrue over decades. A model trained only on quarterly rewards will systematically underweight what matters most.
- **Human-in-the-loop is non-negotiable** for any decision that affects (a) ecological irreversibles, (b) generational beneficiaries, (c) communities not represented at the design table.

## Citations & Further Reading

- Nassim N. Taleb, *Antifragile* (2012) — formal source of the "gain from disorder" concept.
- Kate Raworth, *Doughnut Economics* (2017) — operationalizes ecological + social floors and ceilings.
- Elinor Ostrom, *Governing the Commons* (1990) — multi-stakeholder design principles.
- Long Now Foundation — `https://longnow.org` — for the institutional posture behind century dividends.

---

<sub>Donations: [cash.app/dascient](https://cash.app/dascient/) — supports DaScient, Inc., a non-profit organization aimed to promote accessible intelligence and community learning through various mediums and platforms.</sub>
