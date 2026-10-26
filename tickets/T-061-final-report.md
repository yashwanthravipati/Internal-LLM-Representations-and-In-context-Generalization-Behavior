---
id: T-061
title: Final report
status: todo
phase: 6
priority: P0
depends_on: [T-024]
owner: yashwanth
estimate: L
tags: [writing, course, deliverable]
updated: 2026-10-25
---

# T-061 — Final report

## Context
This is the main course deliverable. Format, length and deadline come from T-001 (update this ticket once known). Start writing as soon as the MVP (T-024) lands; add the Exp 2/3 sections as they finish. Agents can draft sections, but the student owns the final text.

## Tasks
- [ ] Get the format requirements from T-001 and record them here.
- [ ] Outline, following `docs/background/neuro_connections.md` ("Suggested narrative"):
  1. **Abstract.**
  2. **Introduction:** few-shot concept learning in brains (IT geometry, Sorscher); flexible regrouping (Freedman; Mante vs Flesch); global workspace (J-lens); LLM in-context learning as a test bed.
  3. **Methods:** models; tasks and item banks; prompts and controls; representations; the Sorscher theory; shift metrics; the J-lens; statistics and held-out discipline.
  4. **Results:** Exp 1 (H1); Exp 2 (H2a/H2b, H3); Exp 3 (H4, H5); stretch results if any.
  5. **Discussion:** what transfers from neuroscience theory to LLMs and what doesn't; implications for neuroscience (predictions for PFC/IT experiments); limitations (single model family, single-token concepts, the theory's assumptions); nulls stated plainly.
  6. **References.**
- [ ] Every number cites its run directory (in comments or an appendix table). Figures come from T-060.
- [ ] Proofread for claims that go beyond the evidence (decodable ≠ used; correlational ≠ causal unless T-044 was done).

## Acceptance criteria
- [ ] The report meets the course format and is submitted on time.
- [ ] Every figure and number traces to a script and run.

## Outputs
- `report/` (LaTeX or Markdown, per course requirements) and the submitted PDF.

## References
- PLAN; `docs/background/*`; `results/*/REPORT.md`

## Work log
