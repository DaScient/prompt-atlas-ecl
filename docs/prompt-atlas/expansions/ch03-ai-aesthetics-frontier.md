# Chapter 3 · The AI Aesthetics Frontier — Expansion

> **Canonical:** [PROMPT_ATLAS.md#ch3-ai-aesthetics-frontier](../../../PROMPT_ATLAS.md#ch3-ai-aesthetics-frontier) · **Prompts:** [`prompts/ch03.yaml`](../prompts/ch03.yaml) · **Part:** [II](part-ii.md)

## Worked Example — *Adaptive Murals*

**Original prompt:** *"Design public art that responds in real time to weather, air quality, and human voices."*

1. **Choose a wall + sensors** — A public façade wired to temperature, PM2.5, and an opt-in microphone array (with on-device anonymization, no audio leaves the panel).
2. **Define the visual grammar** — A small generative model running locally with a *style anchor* contributed by a community artist; sensor inputs perturb hue, density, motion — never the anchor.
3. **Provenance plate** — A QR plaque names the human anchor artist, the sensor authority, and the model — read in any language.
4. **Veto channel** — A community council can pause, alter, or retire the mural at any time; the council's seat allocation is published.
5. **Sunset rule** — At year five the mural is archived (not deleted) and the wall returns to the community.

## Prompt Templates

```text
# Co-author a piece
"Design a {{form: mural|sculpture|symphony|garden}} co-authored by
 {{human_artist}}, an AI trained on {{specified_canon}}, and
 {{non-human_signal: tide, fungal_network, traffic_pattern}}.
 Specify the provenance plate, the veto channel, and the sunset rule."

# Aesthetics audit
"For {{piece}}, list (1) whose voices are amplified, (2) whose are erased,
 (3) the smallest change that would re-balance the mix without losing the
 anchor artist's intent."
```

## Anti-patterns

- **Style strip-mining.** Using a model to generate "in the style of" a living artist without consent or compensation.
- **Algorithmic uniformity.** Recommendation-driven aesthetics flatten the world even as they multiply images.
- **Beauty as bait.** Awe weaponized for engagement metrics is not awe; it is dependency.
- **Silent provenance.** Any AI-augmented piece without a clear plate of co-authors is laundering authorship.

## Try This

1. **Provenance Plate** — Write the plate for one piece you have made (or might make) with AI assistance.
2. **Canon Audit** — List the dominant cultural canons in your prompt's training set; name three you are *not* drawing from.
3. **Living Mural** — Sketch a one-week trial with a sensor and a local artist.
4. **Veto Channel** — Design the smallest possible community veto for an AI-augmented public artwork.
5. **Sunset Rule** — Decide, in advance, when your AI-augmented piece will be archived.

## Repo Cross-Links

- [`MACP bus`](../../../src/macp/bus.py) — sensor/agent topics for the mural example.
- [`CoLearningMemoryStore`](../../../src/vectorstore/qdrant_store.py) — long-form archives of community-contributed art (with provenance metadata).
- [`server/core_bridge.py`](../../../server/core_bridge.py) — pattern for an opt-in memory toggle (`PAE_MEMORY=1`).

## Guide for AI & Humanity

- **Consent before canon.** Living artists must opt in. Dead artists deserve named attribution.
- **Local before global.** Aesthetics rooted in a specific place beat aesthetics scaled by API.
- **Sensors are political.** A microphone in a plaza is a relationship; design it as one.
- **Reversibility.** Every algorithmic art piece in public space needs an off switch and a sunset clause.

## Citations & Further Reading

- Ursula K. Le Guin, *The Carrier Bag Theory of Fiction* (1986).
- Lewis Hyde, *The Gift* (1983) — the difference between commodity art and gift art.
- Yuk Hui, *Art and Cosmotechnics* (2021).

---

<sub>Donations: [cash.app/dascient](https://cash.app/dascient/) — supports DaScient, Inc., a non-profit organization aimed to promote accessible intelligence and community learning through various mediums and platforms.</sub>
