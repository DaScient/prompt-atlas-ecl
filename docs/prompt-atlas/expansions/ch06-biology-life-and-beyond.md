# Chapter 6 · Biology, Life, and Beyond — Expansion

> **Canonical:** [PROMPT_ATLAS.md#ch6-biology-life-and-beyond](../../../PROMPT_ATLAS.md#ch6-biology-life-and-beyond) · **Prompts:** [`prompts/ch06.yaml`](../prompts/ch06.yaml) · **Part:** [III](part-iii.md)

## Worked Example — *The Coral Whisperer, Done Reversibly*

**Original:** *"Design a synthetic organism whose primary purpose is to restore soil health or ocean ecosystems."* (worked through coral)

1. **Define the smallest viable intervention** — Resist whole-genome edits when a probiotic addition would do.
2. **Build the off switch first** — A kill-switch dependent on a non-natural metabolite *before* anything is released.
3. **Tier the trial** — In silico → micro-reef tank → caged ocean enclosure → small open trial → scale. Each tier requires a documented kill-switch test.
4. **Multispecies consent proxy** — Include marine biologists, indigenous reef stewards, and a "no-release" advocate on the review board.
5. **Reversibility assertion** — Every release has a public, machine-readable rollback plan.

## Prompt Templates

```text
# Reversible bio-design
"Propose the smallest synthetic-biology intervention that would address {{problem}}.
 For the proposed intervention, specify: (1) kill-switch mechanism,
 (2) tiered trial plan with named go/no-go criteria,
 (3) the exact set of stakeholders with veto rights,
 (4) the rollback plan if it escapes containment."

# Interspecies translation skeptic
"Given a claim that {{model}} translates whale/bee/fungal communication,
 list the three most likely confounds, the experiment that would refute them,
 and the human steward who must sign off before any 'translation' is published."

# Pandemic-prediction guardrail
"For a model that forecasts {{viral_property}}, define:
 dual-use risk class, who can access predictions, the redaction policy,
 and the audit trail for any forecast that becomes a deployment trigger."
```

## Anti-patterns

- **Release-then-monitor.** Reversibility must precede release.
- **Single-stakeholder ethics review.** Reef stewards are not optional.
- **In-silico-only validation.** A model that has never met a wet-lab is a model.
- **Dual-use blindness.** Pandemic prediction is also pandemic recipe.

## Try This

1. **Kill-Switch First** — For any bio-design you sketch, write the kill-switch *before* the design.
2. **Tiered Trial Plan** — Translate one of the chapter's prompts into a five-tier go/no-go.
3. **Veto List** — Name three people (not roles) with veto power over your project.
4. **Confound Inventory** — For one "interspecies translation" claim you've seen, list three confounds.
5. **Public Rollback Plan** — One paragraph, machine-readable, for an intervention you'd back.

## Repo Cross-Links

- [`Z3 Tester`](../../../src/testers/z3_tester.py) — encode kill-switch invariants formally.
- [`MACP bus`](../../../src/macp/bus.py) — multi-stakeholder review-board agent topology.
- [`CoLearningMemoryStore`](../../../src/vectorstore/qdrant_store.py) — long-running ecological memory across trial tiers.

## Guide for AI & Humanity

- **Reversibility before novelty.** If you can't undo it, you can't deploy it.
- **Multispecies stakeholders need named human voices.** AI cannot represent the reef; people who depend on it can.
- **Dual-use is not optional.** Every bio-prediction is also a bio-recipe.
- **Stewards over editors.** Co-authoring life is a sacred trust, not a release schedule.

## Citations & Further Reading

- Lynn Margulis, *Symbiotic Planet* (1998).
- Robert Pollack, *Signs of Life* (1994).
- Jennifer Doudna & Sam Sternberg, *A Crack in Creation* (2017).
- Asilomar Conference on Recombinant DNA (1975) — historical model for self-imposed bio-research moratoria.

---

<sub>Donations: [cash.app/dascient](https://cash.app/dascient/) — supports DaScient, Inc., a non-profit organization aimed to promote accessible intelligence and community learning through various mediums and platforms.</sub>
