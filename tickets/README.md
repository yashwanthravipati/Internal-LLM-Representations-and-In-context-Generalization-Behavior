# Ticket system

Plain markdown files, one per work item, managed by `scripts/tickets.py` (Python 3, stdlib only, so it runs without the project environment). `BOARD.md` is generated from the ticket files; never edit it by hand.

## Files

- `T-NNN-<slug>.md`: one ticket. The filename must start with the ticket's `id`.
- `_TEMPLATE.md`: the template used by `tickets.py new`.
- `BOARD.md`: the generated overview (ready queue, active work, per-phase tables).

## Frontmatter fields

```yaml
---
id: T-014                       # T-NNN, unique; numbered in blocks by phase (below)
title: Prompt generator
status: todo                    # todo | in-progress | blocked | review | done | wontfix
phase: 1                        # 0–6, see below
priority: P0                    # P0 = critical path / MVP, P1 = core extension, P2 = stretch
depends_on: [T-010, T-012]      # tickets that must be closed (done/wontfix) first
owner: unassigned               # yashwanth, claude, or another agent/human name
estimate: M                     # S ≤ half a day · M 1–2 days · L 3–5 days
tags: [tasks, data]
updated: 2026-10-25             # set automatically by `tickets.py set`
---
```

### Phases and ID blocks

| Phase | Name | IDs |
|---|---|---|
| 0 | Setup & infrastructure | T-001–T-009 |
| 1 | Task bank & behavior | T-010–T-019 |
| 2 | Neutral geometry & theory (MVP) | T-020–T-029 |
| 3 | In-context geometry & shift | T-030–T-039 |
| 4 | Jacobian lens | T-040–T-049 |
| 5 | Stretch goals | T-050–T-059 |
| 6 | Write-up & deliverables | T-060–T-069 |

`new` takes the first free ID in the phase's block. If the block is full, it uses T-100 and up.

## Body sections

Every ticket has: **Context** (why, linked to PLAN sections), **Tasks** (checklist), **Acceptance criteria** (verifiable), **Outputs** (paths), **Notes & pitfalls**, **References**, and a **Work log** (dated entries appended by whoever works on it).

## Status lifecycle

```
todo ──► in-progress ──► done
   │          │   └────► review ──► done        (needs the user's judgment)
   │          └────────► blocked ──► in-progress
   └───────────────────────────────► wontfix    (out of scope / superseded; say why)
```

- `done`: every acceptance criterion is verified (code run, tests passing, outputs exist).
- `review`: the work is finished, but a scientific or curation judgment needs the user's sign-off (e.g. domain choices, interpreting a result).
- `blocked`: waiting on something external. Explain it in the Work log.
- One agent should have at most one ticket `in-progress` at a time.

## Commands

```bash
python3 scripts/tickets.py ready                       # todo + deps closed, sorted by priority
python3 scripts/tickets.py list [--status S] [--phase N] [--owner O] [--priority P]
python3 scripts/tickets.py show T-014
python3 scripts/tickets.py set T-014 status in-progress owner claude
python3 scripts/tickets.py set T-014 depends_on T-010,T-012
python3 scripts/tickets.py new "Title" --phase 3 --priority P1 --estimate M --depends T-031 --tags geometry
python3 scripts/tickets.py board                       # regenerate BOARD.md (set/new do this too)
python3 scripts/tickets.py check                       # ids match filenames, valid fields, deps exist, no cycles
```

## Conventions

- **Split big tickets** (anything over L) into new tickets, and link them with `depends_on`.
- **New work found mid-ticket** becomes a new ticket. Don't silently grow the scope.
- **Link outputs:** the Work log should point to the `results/<run_id>/` directories, figures and DECISIONS entries the ticket produced.
- **Edit tickets freely** when they're wrong or outdated (the plan will evolve), but note the change in the Work log.
- Run `check` before ending a work session.
