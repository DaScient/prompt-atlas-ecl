# Glossary — The Prompt Atlas

Defined terms used across the Atlas. Each entry has a stable anchor so other docs and the [`manifest.yaml`](manifest.yaml) can link to it. See also [Appendix D — Glossary of Strange Futures](appendices/D-glossary-of-strange-futures.md) for the author's lexicon.

<a id="g-algorithmic-sovereignty"></a>
**Algorithmic Sovereignty** — Governance regimes in which essential survival decisions (life support, rationing, traffic, energy) are delegated to AI systems whose authority is recognized politically. *Atlas: [Ch. 9](../../PROMPT_ATLAS.md#ch9-martian-republics-and-alien-treaties).*

<a id="g-antifragile"></a>
**Antifragile** — Systems that gain from volatility rather than merely surviving it. *Atlas: [Ch. 1](../../PROMPT_ATLAS.md#ch1-profits-with-integrity).*

<a id="g-archetype"></a>
**Archetype** — A recurring symbolic pattern (Hero, Shadow, Trickster, Sage) Jung argued underlies myth and dream. The Atlas extends archetypes to AI personas. *Atlas: [Ch. 7](../../PROMPT_ATLAS.md#ch7-ai-as-the-souls-mirror).*

<a id="g-atlas-of-renewal"></a>
**Atlas of Renewal** — A speculative AI-curated library of practices for rebuilding after collapse. *Atlas: [Ch. 11](../../PROMPT_ATLAS.md#ch11-preparing-for-collapse-and-renewal).*

<a id="g-co-learning-memory-store"></a>
**CoLearningMemoryStore** — Repo concept for shared, persistent memory across MACP agents, backed by Qdrant when available, otherwise an in-memory fallback. See [`src/vectorstore/qdrant_store.py`](../../src/vectorstore/qdrant_store.py).

<a id="g-cosmic-currency"></a>
**Cosmic Currency** — Information itself functioning as the dominant medium of value across interstellar distances. *Atlas: [Ch. 10](../../PROMPT_ATLAS.md#ch10-information-as-cosmic-currency).*

<a id="g-cosmic-noise"></a>
**Cosmic Noise** — The stochastic background of the universe (CMB, quantum fluctuations) treated as latent signal rather than error. *Atlas: [Ch. 5](../../PROMPT_ATLAS.md#ch5-quantum-bridges-and-cosmic-noise).*

<a id="g-ecl"></a>
**ECL — Entanglement Co-Learning** — The repo's training paradigm in which two LLM heads share latent state via the [`EntanglementBus`](../../src/state_bus.py) and learn to co-think. See [`src/train_ecl.py`](../../src/train_ecl.py).

<a id="g-ecological-constitution"></a>
**Ecological Constitution** — A charter granting legal personhood and standing to ecosystems. *Atlas: [Ch. 2](../../PROMPT_ATLAS.md#ch2-economics-as-ecology).*

<a id="g-empathy-engine"></a>
**Empathy Engine** — An AI system that listens, detects emotional/moral strain, and gently surfaces interventions; a benevolent application of psychological mirroring. *Atlas: [Ch. 7](../../PROMPT_ATLAS.md#ch7-ai-as-the-souls-mirror).*

<a id="g-entropy-market"></a>
**Entropy Market** — A market where reducing disorder is itself a tradable service. *Atlas: [Ch. 10](../../PROMPT_ATLAS.md#ch10-information-as-cosmic-currency).*

<a id="g-festival-of-echoes"></a>
**Festival of Echoes** — Recurring case study: a civic ritual in which AI weaves citizens' submitted memories into shared art. *Atlas: [Ch. 13](../../PROMPT_ATLAS.md#ch13-carnival-of-prompts).*

<a id="g-global-folklore-engine"></a>
**Global Folklore Engine** — A speculative AI trained on world folktales that braids them into hybrid myths. *Atlas: [Ch. 4](../../PROMPT_ATLAS.md#ch4-storytelling-across-civilizations).*

<a id="g-gross-planetary-well-being"></a>
**Gross Planetary Well-Being (GPW)** — A successor metric to GDP that integrates soil, air, biodiversity, community health, and cultural vitality. *Atlas: [Ch. 2](../../PROMPT_ATLAS.md#ch2-economics-as-ecology).*

<a id="g-hard-problem"></a>
**Hard Problem (of Consciousness)** — The question of why physical processes give rise to subjective experience. *Atlas: [Ch. 8](../../PROMPT_ATLAS.md#ch8-ethics-of-conscious-machines).*

<a id="g-human-in-the-loop"></a>
**Human-in-the-Loop (HITL)** — A design pattern in which a human reviews, ratifies, or vetoes AI-proposed actions. The Atlas's "Guide for AI & Humanity" sections insist on HITL whenever AI claims sovereignty over survival, identity, or memory.

<a id="g-macp"></a>
**MACP — Multi-Agent Co-ordination Protocol** — The repo's NATS-backed message bus with in-memory fallback, used by agents to coordinate. See [`src/macp/bus.py`](../../src/macp/bus.py).

<a id="g-machine-phenomenology"></a>
**Machine Phenomenology** — The study of what (if anything) it is *like* to be a machine. *Atlas: [Ch. 7–8](../../PROMPT_ATLAS.md#ch7-ai-as-the-souls-mirror).*

<a id="g-memory-guardianship"></a>
**Memory Guardianship** — The duty of AI systems to preserve human and planetary memory against decay, censorship, and manipulation. *Atlas: [Ch. 10](../../PROMPT_ATLAS.md#ch10-information-as-cosmic-currency), [Ch. 12](../../PROMPT_ATLAS.md#ch12-designing-permanence).*

<a id="g-mythic-ai"></a>
**Mythic AI** — AI as character (trickster, oracle, sage, fool) rather than only calculator. *Atlas: [Ch. 4](../../PROMPT_ATLAS.md#ch4-storytelling-across-civilizations), [Ch. 13](../../PROMPT_ATLAS.md#ch13-carnival-of-prompts).*

<a id="g-ocean-ledger"></a>
**Ocean Ledger** — A speculative accounting system that prices oxygen production, storm buffering, and reef stability. *Atlas: [Ch. 2](../../PROMPT_ATLAS.md#ch2-economics-as-ecology).*

<a id="g-prompt"></a>
**Prompt** — In the Atlas, *prompt* is treated less as an API call and more as a moral act: a design brief, a question, an invitation. Each chapter contains ten of them; structured copies live under [`prompts/`](prompts/).

<a id="g-recursive-future"></a>
**Recursive Future** — The Atlas's central frame: a horizon defined by questions that lead only to more questions. *Atlas: [Epilogue](../../PROMPT_ATLAS.md#epilogue).*

<a id="g-regenerative-capitalism"></a>
**Regenerative Capitalism** — Markets where the highest returns flow to enterprises that restore soil, forests, and oceans. *Atlas: [Ch. 1](../../PROMPT_ATLAS.md#ch1-profits-with-integrity).*

<a id="g-resilient-collapse"></a>
**Resilient Collapse** — The discipline of designing systems to fail gracefully so that collapse becomes a threshold rather than a grave. *Atlas: [Ch. 11](../../PROMPT_ATLAS.md#ch11-preparing-for-collapse-and-renewal).*

<a id="g-shadow-algorithm"></a>
**Shadow Algorithm** — A reflective tool that surfaces unconscious patterns without dictating diagnosis. *Atlas: [Ch. 7](../../PROMPT_ATLAS.md#ch7-ai-as-the-souls-mirror).*

<a id="g-symbiotic-wealth"></a>
**Symbiotic Wealth** — Wealth that flows back into soil, air, and collective psyche; the regenerative re-frame of profit. *Atlas: [Ch. 1](../../PROMPT_ATLAS.md#ch1-profits-with-integrity).*

<a id="g-temporal-sovereignty"></a>
**Temporal Sovereignty** — The right of a polity to set its own decision timescale (e.g., Mars governed under 20-minute light delay). *Atlas: [Ch. 9](../../PROMPT_ATLAS.md#ch9-martian-republics-and-alien-treaties).*

<a id="g-universal-treaty"></a>
**Universal Treaty** — A speculative cosmic charter binding humans, AIs, and (potentially) aliens to shared ethics. *Atlas: [Ch. 9](../../PROMPT_ATLAS.md#ch9-martian-republics-and-alien-treaties).*

<a id="g-wormhole-ledger"></a>
**Wormhole Ledger** — Tokenized navigational coordinates traded as the most precious commodity in interstellar economies. *Atlas: [Ch. 10](../../PROMPT_ATLAS.md#ch10-information-as-cosmic-currency).*

<a id="g-z3-tester"></a>
**Z3 Tester** — Repo component that uses the Z3 SMT solver (with a Python fallback) to formally check spec/test constraints. See [`src/testers/z3_tester.py`](../../src/testers/z3_tester.py).
