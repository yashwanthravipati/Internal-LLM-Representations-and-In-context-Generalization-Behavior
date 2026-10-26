---
id: T-006
title: Background reading & notes
status: todo
phase: 0
priority: P1
depends_on: []
owner: yashwanth
estimate: M
tags: [reading, human]
updated: 2026-10-25
---

# T-006 — Background reading & notes

## Context
The report has to make the neuroscience ↔ AI link explicit, and several design choices depend on details in the source papers. The background docs in `docs/background/` are a head start compiled from the sources, but the student should read the key papers directly. Agents can help with summaries.

## Tasks
- [ ] Sorscher, Ganguli & Sompolinsky 2022: main text plus SI §3C (full SNR derivation). Check the normalization conventions against `docs/background/sorscher_theory.md`.
- [ ] Gurnee et al. 2026 (J-lens): at least the method, the J-space definition, the "implicit properties appear when asked" result, the fitting appendix, and limitations.
- [ ] Mante et al. 2013; Flesch et al. 2022; Bernardi et al. 2020 (the H2a/H2b framing and CCGP).
- [ ] Skim: Wei et al. 2023 (flipped/semantically-unrelated labels), Zhao et al. 2021 (calibration), Hendel et al. 2023 / Todd et al. 2024 (task and function vectors: an alternative account of ICL as readout).
- [ ] Update the background docs with anything that changes the plan. Add contradictions with the plan to DECISIONS, and open questions to PLAN §11.

## Acceptance criteria
- [ ] Background docs reviewed and corrected where needed.
- [ ] PLAN §11 (open questions) updated.
- [ ] A short "related work" outline drafted for the report intro (can live in `docs/report_notes.md`).

## Outputs
- Updated `docs/background/*.md`, `docs/report_notes.md`.

## Notes & pitfalls
- The task/function-vector literature matters for interpretation. If ICL works through a compact "task vector" at the decision position, that favors H2b (readout selection).

## References
- PLAN §12; `docs/background/neuro_connections.md`

## Work log
