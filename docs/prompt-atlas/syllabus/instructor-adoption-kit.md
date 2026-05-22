# Instructor Adoption Kit — *The Prompt Atlas: Twelve Quests*

> **Audience:** instructors, department chairs, curriculum directors, district AI coordinators, corporate L&D leads.
> **Goal:** make the syllabus drop-in adoptable in one planning session.

---

## 1. Pacing Options

Pick the row that matches your calendar. All three deliver the same 12 quests and the same capstone.

| Format | Per-week load | Total contact hrs | Best for |
|---|---|---|---|
| **Semester (default)** — 12 wks + Festival + Capstone | 3–4 hrs | ~50 | Undergrad, grad seminar |
| **Half-load (24 wks)** — one quest every two weeks | 1.5–2 hrs | ~50 | High school year-long elective |
| **Bootcamp (6 wks)** — two quests per week, paired by Part | 6–8 hrs | ~45 | Summer intensive, corporate L&D, faculty PD cohort |
| **Quarter (10 wks)** — collapse Wks 5+6 and Wks 11+12 | 3–4 hrs | ~40 | Quarter-system colleges |
| **Micro-credential (4 wks)** — one Part per week, learner picks one quest per Part | 3 hrs | ~12 | Open-enrollment continuing ed; certificate stack |

**Pairing rules for the Bootcamp:**
- Day 1 of each week: do the *Premise + Field Notes + Co-Creation* of both paired quests.
- Day 2: cross-pollinated *Field Test* (apply Quest A's artifact to Quest B's domain). This makes LO4 (interdisciplinary transfer) explicit.

---

## 2. Standards Alignment Matrix

> *Non-exhaustive — designed to give you defensible language for accreditation reports.*

### ISTE Standards for Students (2016, reaffirmed 2024)

| ISTE | Where it shows up in the syllabus |
|---|---|
| **1 Empowered Learner** | Capstone portfolio (student-controlled hosting); Thirteenth Quest |
| **2 Digital Citizen** | Every weekly provenance plate; Week 8 Treaty; Week 10 Information Diet |
| **3 Knowledge Constructor** | Weeks 1, 2, 5, 10 (data + source triangulation against AI) |
| **4 Innovative Designer** | Weeks 3, 6, 9, 12 (designing with constraints, sunsets, durability) |
| **5 Computational Thinker** | Weeks 2, 5, 11 (modeling, residuals, failure modes) |
| **6 Creative Communicator** | Weeks 3, 4, 7 (myth, translation, witnessing) |
| **7 Global Collaborator** | Week 4 (cross-language); Week 9 (off-world polity); Festival |

### AAC&U VALUE Rubrics

| VALUE rubric | Quest evidence |
|---|---|
| Critical Thinking | Council reviews, Week 5 residuals, Week 1 ledger |
| Ethical Reasoning | Every provenance plate; Week 8 Treaty; Week 7 Mirror |
| Information Literacy | Week 10 Information-as-Salt; Week 4 Translator |
| Integrative Learning | Capstone Recursive Reflection + Thirteenth Quest |
| Civic Engagement | Week 1 Field Test; Week 9 Charter; Week 11 Collapse Drill |
| Inquiry & Analysis | Week 5; Week 6; Week 2 supply-chain interview |
| Creative Thinking | Week 3; Week 6; Week 12; Festival |
| Written Communication | All weekly reflections; capstone defense memo |
| Oral Communication | Capstone defense; Festival booth |
| Intercultural Knowledge | Week 4; Week 9; Week 3 canon audit |

### NGSS Cross-Cutting Concepts

| NGSS CCC | Quest |
|---|---|
| Systems & system models | Wk 2 Economy as Forest, Wk 6 Living Prompt |
| Cause & effect | Wk 1 Honest Ledger, Wk 11 Collapse Drill |
| Patterns | Wk 5 Listening to Noise |
| Stability & change | Wk 12 Designing Permanence |

### OECD AI Literacy Framework (draft, 2024) & UNESCO AI Ethics Recommendation (2021)

Both frameworks emphasize *responsible use, human oversight, transparency, and inclusion.* The course's **Guide for AI & Humanity Rubric** (5 axes: Consent, Locality, Provenance, Reversibility, Honesty) is a direct operationalization of these principles. Map it line-by-line during your local approval process.

---

## 3. Assessment Logistics

### Suggested LMS gradebook setup

```
Category                          Weight   Items
─────────────────────────────────────────────────
Weekly Quest Artifacts             36%      12 × 3%
Weekly Reflections                 12%      12 × 1%
Council Participation              12%      12 × 1%
Thirteenth Quest Design Memo       10%      1
Capstone Portfolio                 20%      1
Capstone Defense                   10%      1
─────────────────────────────────────────────────
Total                             100%
```

### Provenance Plate — machine-readable schema (paste into a YAML front-matter)

```yaml
plate:
  artifact: "Origin Myth of the Missing Sock"
  week: 3
  human_authors: ["Student Name"]
  ai_co_authors:
    - model: "<vendor / model / version>"
      role: "draft continuations, canon audit"
      session_date: "2026-09-17"
  data_sources: []
  canons_drawn_from: ["Norse", "Yoruba (public folklore)"]
  canons_excluded: ["sacred Lakota stories — closed tradition"]
  veto_channel: "instructor + 2 peers"
  sunset_date: "2027-06-01"
  signature: "Student Name, 2026-09-17"
```

Grading the plate is *part of grading the artifact*. A missing or false plate is the failure mode, not AI use itself.

### One-sheet Council rubric (print double-sided)

```
        Axis            0           1           2           3
        Consent         _____       _____       _____       _____
        Locality        _____       _____       _____       _____
        Provenance      _____       _____       _____       _____
        Reversibility   _____       _____       _____       _____
        Honesty         _____       _____       _____       _____

        Reviewer:                   Artifact:                Date:
        One specific compliment:
        One specific concern:
        One concrete next step:
```

Three reviewers per artifact. Median wins.

---

## 4. Tool-Stack Suggestions (all optional, all swappable)

| Need | Free option | Paid alternative | Analog parallel |
|---|---|---|---|
| Chat-LLM | any modern free-tier chat assistant | paid-tier same | paper + classroom shared session |
| Versioning | Git + GitHub Pages | GitHub Pro | bound paper portfolio |
| Diagramming | Mermaid in Markdown | Figma | pencil & graph paper |
| Translation cross-check (Wk 4) | any free MT + native-speaker check | n/a | community language partner |
| Local AI for Wk 7 (privacy) | open-weights model on a school laptop | n/a | journaling without AI; reflect on the difference |

> **No student is required to create an account on any platform whose ToS conflicts with your district policy.** A class-shared session run by the instructor is always an acceptable substitute.

---

## 5. Common Objections (and answers)

> **"Won't students just have the AI do everything?"**
> The artifact is not the grade. The **provenance plate, the Council review, and the reflection** are the grade. A student who outsources cannot fake the plate without fabricating evidence — and fabricating evidence is the standard, well-understood violation we already know how to handle.

> **"Our AUP forbids AI use."**
> Then use the syllabus's paper-and-pencil parallels for the co-creation step, and have students annotate *publicly visible* AI outputs from a teacher-controlled session. The questions, audits, and ethics framing remain unchanged.

> **"We don't have AI tool budget."**
> Total dollar cost to adopt this syllabus is **zero**. All texts, prompts, expansions, rubrics, and tools have a free path.

> **"Our parents are worried about AI."**
> Send them the [`pitch-to-curriculum-directors.md`](pitch-to-curriculum-directors.md) and Week 7's privacy footnote. The course's posture is **not** "AI is exciting!" It is "AI is here; here is how we read it, audit it, and refuse it when necessary."

> **"How do we assess hallucination?"**
> Weeks 4, 5, and 6 *require* students to document where the AI was wrong. Errors are an assessment surface, not a failure.

> **"Is this 'AI in the classroom' or 'classroom about AI'?"**
> Both. By design. Students *use* AI in 11 of 12 quests, *while* studying its limits in all 12.

---

## 6. Adopter Checklist (one planning session)

- [ ] Pick a pacing format from §1.
- [ ] Confirm AUP path (commercial AI? shared classroom session? open-weights local?).
- [ ] Decide hosting for student portfolios (GitHub Pages by default).
- [ ] Print 12 Quest Cards from [`weekly-quest-cards.md`](weekly-quest-cards.md) for classroom walls.
- [ ] Add the standards-alignment language from §2 to your course proposal.
- [ ] Schedule Festival Week date and invite guests early.
- [ ] Open an issue at [`DaScient/prompt-atlas-ecl`](https://github.com/DaScient/prompt-atlas-ecl/issues) to be added to the adoption wall.

---

## 7. FAQ

**Q. Can I sell my remix?**
A. Per the repository license — adapt, translate, and use freely. Don't repackage the canonical Atlas prose as your own; cite per [`atlas_respect.md`](../../../atlas_respect.md).

**Q. Can I substitute a chapter?**
A. Yes. Chapters 13–14 (*Carnival*, *Wonder*) are designed as substitution candidates — they can replace any earlier quest if your context calls for it.

**Q. Does this work in a non-English-speaking program?**
A. The structure is language-agnostic. Week 4 (*Translator's Oath*) becomes even richer in multilingual cohorts. We welcome translations as PRs.

**Q. Is there a high-school version?**
A. Yes — use the *Half-load (24 wks)* pacing and the paper-and-pencil parallel for Week 7. We're collecting Grade 9–12 teacher remixes; tag your issue `k12-remix`.

**Q. Does this work for corporate L&D?**
A. Yes — the Bootcamp pacing is designed for it. Replace the Capstone Portfolio's "public hosting" requirement with "internal wiki" if needed.

---

*Maintained by DaScient Press, Ltd. · Pull requests welcome.*
