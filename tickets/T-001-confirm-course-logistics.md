---
id: T-001
title: Confirm course logistics, deadlines & deliverable format
status: todo
phase: 0
priority: P0
depends_on: []
owner: yashwanth
estimate: S
tags: [course, human]
updated: 2026-10-25
---

# T-001 — Confirm course logistics, deadlines & deliverable format

## Context
The project-list PDF (`docs/course/Projects_List_2026.pdf`) says which projects are allowed, but not the deadlines, deliverable format, team rules or compute. Planning depends on these: the timeline (PLAN §9), the report and presentation tickets (T-061, T-062), the proposal (T-007), and the model/compute choices (T-003). **This is a human ticket:** only the student can get these answers from the syllabus or course staff.

## Tasks
- [ ] Find the final-project deadline and any intermediate milestones (proposal, check-in, presentation dates).
- [ ] Find the required deliverables: report format and length, code submission, presentation or poster.
- [ ] Confirm team size (solo vs group) and, for a group, who owns which parts.
- [ ] Confirm the compute available: course cluster? GPU type and VRAM? `/scratch` quota? Colab? (Other projects on the list mention "one 16 GB GPU" and `/scratch`.)
- [ ] Confirm that extending Project 5 (context-dependent regrouping + Jacobian lens) needs no extra approval. If unsure, check with the instructors.
- [ ] Confirm the project doesn't overlap with a project in another class or lab (a course rule).
- [ ] Fill in the Logistics table in `docs/course/COURSE_CONTEXT.md`, and the deadlines line in `CLAUDE.md` ("Deadlines: TBD").

## Acceptance criteria
- [ ] The COURSE_CONTEXT.md logistics table has no TBD rows (or the remaining ones are explicitly unknowable for now).
- [ ] T-061 / T-062 updated with format and length requirements.
- [ ] Milestone dates added to `docs/PLAN.md` §9, with phase targets back-planned from the deadline.

## Outputs
- Updated `docs/course/COURSE_CONTEXT.md`, `docs/PLAN.md` §9, `CLAUDE.md`.

## Notes & pitfalls
- Back-plan from the deadline so the MVP (T-024, milestone M2) lands with about 40–50% of the schedule still left.
- If compute is tighter than 16 GB, the main model drops to Qwen3.5-0.8B/2B. Tell whoever owns T-003/T-004.

## References
- `docs/course/COURSE_CONTEXT.md`, PLAN §9

## Work log
