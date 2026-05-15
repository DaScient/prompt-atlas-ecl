# Part I — Prosperity and Purpose · Expansion

> **Diagram:** [`diagrams/part-i-prosperity.md`](../diagrams/part-i-prosperity.md)
> **Chapters:** [Ch. 1 · Profits with Integrity](../../../PROMPT_ATLAS.md#ch1-profits-with-integrity), [Ch. 2 · Economics as Ecology](../../../PROMPT_ATLAS.md#ch2-economics-as-ecology)

## Why this Part exists

Part I asks the question that gates everything else: *what kind of wealth do we want AI to produce?* If profit remains a single number on a quarterly report, AI will optimize the world into something brittle. The two chapters argue for **symbiotic wealth** (Ch. 1) and **economics as ecology** (Ch. 2) — a re-framing that makes integrity a non-negotiable input, not a moral footnote.

## Through-lines

- **From extraction to regeneration** — both chapters reject the "growth is enough" thesis and ask AI to optimize for regeneration rates, not just extraction rates.
- **From quarter to century** — *century dividends* (Ch. 1) and *living constitutions* (Ch. 12, foreshadowed) reframe time horizons.
- **From human-only to multispecies stakeholders** — rivers, forests, oceans become parties to the contract (Ch. 2's *Ocean Ledger*, *river-as-shareholder*).

## How this Part connects to the repo

| Atlas concept | Repo touch-point |
|---|---|
| AI auditor for integrity ([Ch. 1](../../../PROMPT_ATLAS.md#ch1-profits-with-integrity)) | [`Z3 Tester`](../../../src/testers/z3_tester.py) — formal checks with Python fallback |
| Multi-stakeholder coordination ([Ch. 2](../../../PROMPT_ATLAS.md#ch2-economics-as-ecology)) | [`MACP bus`](../../../src/macp/bus.py) — agent coordination with NATS or in-memory fallback |
| Tracking long horizons | [`MLflow tracker`](../../../src/tracking/__init__.py) (`PAE_TRACKING=1`) |

## Reading order

1. Sit with the chapter prose first.
2. Open the per-chapter expansions for templates and exercises:
   - [`ch01-profits-with-integrity.md`](ch01-profits-with-integrity.md)
   - [`ch02-economics-as-ecology.md`](ch02-economics-as-ecology.md)
3. Pull the prompts as YAML from [`prompts/ch01.yaml`](../prompts/ch01.yaml) and [`prompts/ch02.yaml`](../prompts/ch02.yaml).

## Guide for AI & Humanity (Part-level)

- **Human-in-the-loop** — Any AI assigned to the role of "integrity auditor" or "ecological mediator" must surface its judgments to a human ratifier. Audit trails are non-optional.
- **Beware proxy metrics** — Optimizing a single composite "regeneration index" can be gamed in the same ways GDP was. Carry several measures (carbon, biodiversity, equity) without collapsing them prematurely.
- **Don't outsource conscience** — AI can model, monitor, and warn. It cannot decide what *deserves* to flourish.
