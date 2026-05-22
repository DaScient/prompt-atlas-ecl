# The Prompt Atlas — 12-Week Quest Syllabus

> **For curriculum directors, deans of innovation, learning designers, and bold high-school department heads.**
> A free, open, twelve-week course that turns *The Prompt Atlas — Kronos Edition* into a **questline** for the recursive age of AI.

| | |
|---|---|
| **Course title** | *The Prompt Atlas: Twelve Quests for the Recursive Age* |
| **Length** | 12 weeks · 3–4 hrs / week (flexible: bootcamp, semester, micro-credential) |
| **Audience** | Upper-secondary (Grades 11–12), undergraduate, graduate seminar, professional development |
| **Cost to adopt** | **$0.** Free under the repository license (see [LICENSE](../../../LICENSE) + [atlas_respect.md](../../../atlas_respect.md)) |
| **Format** | Markdown · printable · PDF-ready · LMS-importable |
| **Companion text** | [`PROMPT_ATLAS.md`](../../../PROMPT_ATLAS.md) (Kronos Edition, 2026) — also free |
| **Status** | v1.0 · maintained alongside the Atlas |

---

## 📥 Download & Use

Everything in this folder is plain Markdown — no paywall, no login, no email gate.

| File | Use it for |
|---|---|
| **[`12-week-quest-syllabus.md`](12-week-quest-syllabus.md)** | The full syllabus — adopt as-is, or remix |
| **[`instructor-adoption-kit.md`](instructor-adoption-kit.md)** | Pacing options, standards alignment, assessment matrices, FAQ |
| **[`weekly-quest-cards.md`](weekly-quest-cards.md)** | Twelve printable one-page **Quest Cards** — hand them out, pin them on classroom walls |
| **[`pitch-to-curriculum-directors.md`](pitch-to-curriculum-directors.md)** | A one-page memo you can forward to your dean, principal, or board |

### Convert to PDF or DOCX (one-liners)

```bash
# Combined PDF of the whole syllabus pack
pandoc README.md 12-week-quest-syllabus.md instructor-adoption-kit.md \
       weekly-quest-cards.md pitch-to-curriculum-directors.md \
       -o PromptAtlas-12Week-Syllabus.pdf --toc --pdf-engine=xelatex

# Editable DOCX for your district's template
pandoc 12-week-quest-syllabus.md -o PromptAtlas-12Week-Syllabus.docx
```

### Import into your LMS

- **Canvas / Moodle / Blackboard / Brightspace:** each week is a self-contained module — copy the Markdown into a page or convert with `pandoc --to=html`.
- **Notion / Coda / Confluence:** paste directly; tables and headings preserve.
- **Google Classroom:** each Quest Card prints to a single sheet — distribute weekly.
- **GitHub Classroom:** fork this repo; each week already has a `Try This` block ready to be turned into an assignment.

---

## 🧭 What is a "quest"?

A **quest** is one week of structured exploration anchored in a single Atlas chapter. Every quest has the same six beats — students always know what to expect, while the *content* changes radically week to week:

1. **Premise** — a one-paragraph mythic framing of the week's question.
2. **Field Notes** — the canonical reading from `PROMPT_ATLAS.md` + the chapter expansion.
3. **Co-Creation** — a hands-on AI exercise (the user's example: *Week 3 — co-write a myth with an AI*).
4. **Field Test** — apply the result in a real or simulated public context.
5. **Council** — peer + ethics review using the **Guide for AI & Humanity** rubric.
6. **Logbook** — a portfolio artifact, signed and dated, with a *provenance plate*.

The full week-by-week breakdown lives in [`12-week-quest-syllabus.md`](12-week-quest-syllabus.md).

---

## 🎯 Why a curriculum director should care

> **AI literacy is no longer an elective. It is the new civics.**

This course gives you a *ready-made, free, open-licensed* answer to four questions every accreditation review is now asking:

1. **"How are you teaching responsible AI use?"** — Every week ends with an explicit ethics council and a provenance plate.
2. **"How are you preventing AI-assisted cheating?"** — We don't ban AI; we make it the **co-author of record**. Assessment is about authorship, judgment, and provenance — not output.
3. **"How does this map to standards?"** — See the alignment matrix in [`instructor-adoption-kit.md`](instructor-adoption-kit.md): ISTE Standards for Students, AAC&U VALUE rubrics, NGSS cross-cutting concepts, OECD AI Literacy Framework (draft).
4. **"What does the capstone show employers / admissions?"** — A public **Atlas Portfolio**: 12 signed quests + 1 capstone, hosted on the student's own GitHub or domain. Portable. Verifiable. Theirs forever.

A two-page version of this argument lives in [`pitch-to-curriculum-directors.md`](pitch-to-curriculum-directors.md) — forward it to your dean.

---

## 🪶 Pedagogical lineage

This syllabus stands on three traditions:

- **Quest-based learning** (Gee, Salen Tekinbaş) — turn a course into a structured adventure with a visible questline.
- **Carrier-bag pedagogy** (Le Guin) — the course is a *container* for things students gather; it is not a hero's-journey gauntlet.
- **The Atlas's own "recursive" framing** — every chapter returns to its themes. So does every week. Mastery is spiral, not linear.

---

## 🛡️ Accessibility, ethics, and academic integrity

- **Accessibility.** Every quest is delivered in plain Markdown (screen-reader friendly), with alt text on diagrams and no required proprietary tools. A "low-tech" path is offered for every co-creation activity — pen, paper, and a single shared classroom AI session suffice.
- **Ethics.** Every week explicitly invokes the **Guide for AI & Humanity** notes from the chapter expansion. Consent, attribution, sunset rules, and veto channels are *graded*, not optional.
- **Integrity.** Students do not hide AI use — they **document** it on a *provenance plate* attached to each artifact. The plate is the assignment.
- **Data minimization.** Suggested AI tools are listed with privacy notes; no student work is required to be uploaded to a third-party platform.

---

## 🤝 How to contribute / remix

This syllabus is part of [`DaScient/prompt-atlas-ecl`](https://github.com/DaScient/prompt-atlas-ecl) and the broader [`DaScient/The-Prompt-Atlas`](https://github.com/DaScient/The-Prompt-Atlas) project.

- **Fork it.** Adapt for your district, college, or company L&D program.
- **Open an issue** to share a remix, propose a new quest, or flag an ethics concern.
- **Pull-request improvements** to the rubrics, alignment matrix, or accessibility notes.

> *"Dwell on the beauty of life. Watch the stars, and see yourself running with them."* — algoritmi, quoted in the Atlas front matter.

---

*Maintained by DaScient Press, Ltd. · Released free, in perpetuity, for the leaders of the recursive age.*
