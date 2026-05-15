# Chapter 4 · Storytelling Across Civilizations — Expansion

> **Canonical:** [PROMPT_ATLAS.md#ch4-storytelling-across-civilizations](../../../PROMPT_ATLAS.md#ch4-storytelling-across-civilizations) · **Prompts:** [`prompts/ch04.yaml`](../prompts/ch04.yaml) · **Part:** [II](part-ii.md)

## Worked Example — *The Global Folklore Engine, Done Honestly*

**Original:** *"Merge the Epic of Gilgamesh, Buddhist sutras, and African griot tales into a single narrative about survival in the AI age."*

1. **Curate the canon with the carriers** — For each tradition, name a living tradent or scholar who has consented to contribute and who will hold a veto.
2. **Embed structurally, not stylistically** — Capture the *story-grammars* (descent, return, trickster reversal) rather than imitating prose surfaces. This avoids pastiche.
3. **Generate as scaffold** — AI proposes a braid; humans (the named carriers) revise, reject, or re-anchor.
4. **Mark synthesis** — The published myth is labeled as a 21st-century *braid*, not as discovered tradition.
5. **Royalty stream** — Each tradition's tradent receives perpetual attribution and revenue share.

## Prompt Templates

```text
# Honest mythography
"From the named traditions {{T1, T2, T3}} (consented carriers: {{names}}),
 extract three shared story-grammars. Propose one braid that uses all three,
 marked as a 21st-century synthesis with attribution to each carrier."

# Failure-utopia
"Describe a utopia designed by an AI optimizing {{single_metric}}.
 Walk it forward 200 years until it collapses. Identify the moment the
 metric and the meaning diverged. Name one early warning sign."

# Synthetic-myth audit
"Given {{narrative}} circulating in {{network}}, return:
 (1) traceable provenance, (2) which audiences it consoles vs. mobilizes,
 (3) a counter-narrative drawn from a tradition NOT represented here."
```

## Anti-patterns

- **Pastiche-as-canon.** Generating "in the style of" without consent flattens the world.
- **Recommendation-driven myth.** Optimizing narrative for engagement guarantees it will exploit fear.
- **Single-source training.** A model fed by one civilization will speak as that civilization while pretending to be universal.
- **Erased provenance.** Synthetic myths without attribution are propaganda waiting to be discovered.

## Try This

1. **Story-Grammar Walk** — Pick a story you love. Extract its *grammar* (not its prose). Re-tell it from a different cultural anchor.
2. **Failure-Utopia** — Use the template above on a metric your industry currently optimizes.
3. **Carrier Map** — For your favorite myth, name a *living* carrier. Reach out.
4. **Synthetic-myth Audit** — Take one viral narrative and run the audit prompt.
5. **Cross-Species Fable** — Draft a one-page fable co-authored with a non-human signal (river data, bee waggle dance).

## Repo Cross-Links

- [`MACP bus`](../../../src/macp/bus.py) — multi-tradent agents passing drafts.
- [`CoLearningMemoryStore`](../../../src/vectorstore/qdrant_store.py) — provenance-tagged narrative archive.
- [`Z3 Tester`](../../../src/testers/z3_tester.py) — encode invariants ("attribution must accompany any descendant of this story") as constraints.

## Guide for AI & Humanity

- **Consent before canon — again.** Tradents are not training data.
- **Mark synthesis.** Honest braids beat undisclosed ones.
- **Counter-weight minorities.** When AI compresses, statistical minorities vanish first.
- **Honor failure stories.** Fables of collapse are vaccines.

## Citations & Further Reading

- Joseph Campbell, *The Hero with a Thousand Faces* (1949) — story-grammar source.
- Marie-Louise von Franz, *The Interpretation of Fairy Tales* (1970).
- Ngũgĩ wa Thiong'o, *Decolonising the Mind* (1986).
- The World Oral Literature Project — `https://www.oralliterature.org/`.
