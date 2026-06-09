# Chapter 2 · Economics as Ecology — Expansion

> **Canonical:** [PROMPT_ATLAS.md#ch2-economics-as-ecology](../../../PROMPT_ATLAS.md#ch2-economics-as-ecology) · **Prompts:** [`prompts/ch02.yaml`](../prompts/ch02.yaml) · **Part:** [I](part-i.md)

## Worked Example — *The River as Shareholder*

**Original prompt:** *"Simulate a trade system where oceans are represented as sovereign entities negotiating extraction rights."* (adapted to a river for tractability)

1. **Define the legal person** — A watershed (e.g., the Ganges, Whanganui — already real) with a charter, trustees, and an AI-monitored set of physical metrics (flow, dissolved O₂, sediment balance, fish counts).
2. **Issue *river dividends*** — Any entity drawing water owes dividends *back to the river*: oxygen production, sediment restoration, fish-population recovery — measured by sensor networks and audited monthly.
3. **Pricing** — Dividends are denominated in *ecological units* (mg/L of dissolved O₂, kg/m³ of sediment, fish/km³). A market emerges where ecological units are tradable but never *retired* without river-trustee consent.
4. **Investor logic** — A fund invests in upstream restoration; if river metrics improve, dividends are paid; if metrics worsen, the fund's stake is diluted. Investor ROI is now physically coupled to ecological outcomes.
5. **Failure modes** — Sensor capture, jurisdictional gaps, and "metric laundering" all need explicit countermeasures (open sensors, multi-jurisdictional trust, periodic dossier review).

## Prompt Templates

```text
# Externality reveal
"For {{product_or_policy}}, list every cost currently treated as an externality.
 For each, propose a sensor or proxy that would internalize it,
 and the human reviewer who would adjudicate disputes."

# Ocean Ledger probe
"Treat {{coastal_region}} as if its ocean services (oxygen, storm buffering,
 climate regulation, fisheries) were line items on a national balance sheet.
 Restate last year's regional GDP with these line items debited or credited
 by ecological change."

# Rights-of-nature draft
"Draft a one-page charter granting legal personhood to {{ecosystem}}.
 Include: trustees, AI-monitored metrics, dispute resolution,
 and the rule for when the charter itself can be amended."
```

## Anti-patterns

- **Ecology theatre.** Naming a river "person" without a sensor network and enforcement is a press release, not a policy.
- **Single-metric capture.** A river measured only by flow can be drained of life while looking healthy.
- **AI as final arbiter.** Algorithmic adjudication of ecological disputes without human appeal repeats colonial patterns under new branding.
- **GDP-with-extras.** Bolting "well-being" onto GDP is not the same as designing a successor metric.

## Try This

1. **Externality Map** — For one product your org sells, list five externalities and an internalization strategy for each.
2. **Local Ocean Ledger** — Restate one budget line in your municipality with an ecological line item.
3. **Charter Draft** — One page granting personhood to a local ecosystem. Share it with someone who would *implement* it.
4. **Sensor Walk** — Walk through your supply chain and mark each step where you currently rely on *attestation* rather than *measurement*.
5. **Conflict Rehearsal** — Pick two ecological metrics that pull against each other (e.g., dam-flow vs. fish migration). Write the rule for adjudication.

## Repo Cross-Links

- [`MACP bus`](../../../src/macp/bus.py) — multiple "stakeholder" agents (river, industry, municipality) coordinating via topics; in-memory fallback means you can prototype without infra.
- [`CoLearningMemoryStore`](../../../src/vectorstore/qdrant_store.py) — long-running ecological memory (decade-scale river history) embedded for retrieval.
- [`Z3 Tester`](../../../src/testers/z3_tester.py) — encode the river charter's invariants (e.g., "annual dividend ≥ minimum O₂ floor") as constraints.

## Guide for AI & Humanity

- **Sensors > attestations.** Where physical measurement is possible, prefer it. Where impossible, name the human accountable.
- **Multispecies stakeholders need human translators.** AI can model the river's voice; only people can be sued in court.
- **Slow time wins.** Ecological dividends operate on decade scales; reward the systems that wait.
- **Refuse the false dichotomy** of "growth vs. environment." The successor metric is *continuity* — the integral of well-being over time.

## Citations & Further Reading

- Te Awa Tupua (Whanganui River Claims Settlement) Act 2017, New Zealand — actual rights-of-nature precedent.
- Herman Daly, *Beyond Growth* (1996).
- Donella Meadows, *Thinking in Systems* (2008) — leverage points and feedback loops.
- IPCC AR6 — for the physical baseline against which ecological dividends must be denominated.

---

<sub>Donations: [cash.app/dascient](https://cash.app/dascient/) — supports DaScient, Inc., a non-profit organization aimed to promote accessible intelligence and community learning through various mediums and platforms.</sub>
