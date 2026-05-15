# `docs/prompt-atlas/` — Companion Materials

Companion scaffolding for the canonical [`PROMPT_ATLAS.md`](../../PROMPT_ATLAS.md) (Kronos Edition).

The canonical file holds the **author's prose** (lightly edited for clarity). This directory holds everything *around* it that makes the Atlas usable as a guide for both AI and humanity:

```
docs/prompt-atlas/
├── README.md                ← you are here
├── manifest.yaml            ← machine-readable index of every part, chapter, anchor, prompt
├── glossary.md              ← defined terms with anchors
├── index.md                 ← thematic cross-references (ethics, ecology, myth, …)
├── diagrams/                ← one Mermaid diagram per Part
│   ├── part-i-prosperity.md
│   ├── part-ii-culture.md
│   ├── part-iii-science.md
│   └── part-iv-psyche.md
├── prompts/                 ← extracted prompts as structured YAML
│   ├── ch01.yaml … ch07.yaml
└── expansions/              ← per-chapter & per-part expansions
    ├── part-i.md … part-iv.md
    └── ch01-…md … ch07-…md
```

Each chapter expansion follows a consistent template:

| Section | Purpose |
|---|---|
| **Worked Examples** | Concrete walk-throughs of one or two prompts end-to-end |
| **Prompt Templates** | Reusable prompts with `{{variables}}` and fill-in guidance |
| **Anti-Patterns** | What goes wrong if the prompt is asked carelessly |
| **Try This** | 3–5 hands-on exercises a reader can do today |
| **Repo Cross-Links** | Where the chapter's themes already live in this codebase (MACP bus, `CoLearningMemoryStore`, ECL trainer, geometry losses, Z3 tester, agents, …) |
| **Guide for AI & Humanity** | Ethics, safety, human-in-the-loop notes — the explicit framing this Atlas adds |
| **Citations & Further Reading** | Pointers for deeper study |

## How agents consume this

The [`manifest.yaml`](manifest.yaml) is the canonical machine entry-point. It lists every section anchor in `PROMPT_ATLAS.md` and every prompt file under `prompts/`, so an agent can:

1. Load the manifest to discover available chapters.
2. Read structured prompt YAMLs into its prompt registry.
3. Optionally embed prose chunks into the [`CoLearningMemoryStore`](../../src/vectorstore/qdrant_store.py) for retrieval-augmented dialogue (enable with `PAE_MEMORY=1`; see [`server/core_bridge.py`](../../server/core_bridge.py)).
4. Resolve glossary terms via [`glossary.md`](glossary.md) anchors.

A minimal example loader lives in the manifest's top-level comments.

## Editorial principles

- The author's voice is preserved. Edits are limited to: typo fixes, smart-quote/dash normalization, bullet-marker consistency, removal of trivially duplicated paragraphs, and consistent heading levels for anchorability.
- All prompts are **reproduced verbatim from the prose** in `prompts/chNN.yaml`. Templates and worked examples in `expansions/` are *additive* and clearly labeled as such.
- The "Guide for AI & Humanity" framing adds ethics/safety notes; it never overrides the author's claims.

For attribution and respect for the source work, see [`atlas_respect.md`](../../atlas_respect.md).
