#!/usr/bin/env python3
"""File-based ticket tracker for this repo (stdlib only, no env needed).

Each ticket is tickets/T-NNN-<slug>.md with a frontmatter block between two
'---' lines. See tickets/README.md for field meanings and workflow.

Commands:
  list   [--status S] [--phase N] [--owner O] [--priority P]   table of tickets
  ready                                    todo tickets whose deps are all closed
  show   ID                                print a ticket
  set    ID KEY VALUE [KEY VALUE ...]      edit frontmatter (status, owner, ...)
  new    TITLE --phase N [--priority P1] [--estimate M] [--depends T-1,T-2] [--tags a,b]
  board                                    regenerate tickets/BOARD.md
  check                                    validate all tickets (ids, deps, cycles)

IDs may be written as T-014, t-14 or 14. `set` and `new` regenerate the board.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TICKET_DIR = ROOT / "tickets"
BOARD = TICKET_DIR / "BOARD.md"
TEMPLATE = TICKET_DIR / "_TEMPLATE.md"

STATUSES = ["todo", "in-progress", "blocked", "review", "done", "wontfix"]
CLOSED = {"done", "wontfix"}
PRIORITIES = ["P0", "P1", "P2"]
ESTIMATES = ["S", "M", "L"]
PHASES = {
    0: "Setup & infrastructure",
    1: "Task bank & behavior",
    2: "Neutral geometry & theory (MVP)",
    3: "In-context geometry & shift",
    4: "Jacobian lens",
    5: "Stretch goals",
    6: "Write-up & deliverables",
}
LIST_FIELDS = {"depends_on", "tags"}
FM_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)


def normalize_id(raw: str) -> str:
    m = re.fullmatch(r"(?:[Tt]-?)?(\d+)", raw.strip())
    if not m:
        sys.exit(f"error: bad ticket id {raw!r}")
    return f"T-{int(m.group(1)):03d}"


def id_num(tid: str) -> int:
    return int(tid.split("-")[1])


def parse_value(key: str, raw: str):
    raw = raw.strip()
    if key in LIST_FIELDS:
        return [x.strip() for x in raw.strip("[]").split(",") if x.strip()]
    if key == "phase":
        return int(raw)
    return raw.strip("\"'")


def load(path: Path) -> dict:
    m = FM_RE.match(path.read_text())
    if not m:
        sys.exit(f"error: {path.name} has no frontmatter")
    meta = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, raw = line.partition(":")
        meta[key.strip()] = parse_value(key.strip(), raw)
    meta["path"] = path
    return meta


def load_all() -> list[dict]:
    tickets = [load(p) for p in TICKET_DIR.glob("T-*.md")]
    return sorted(tickets, key=lambda t: id_num(t["id"]))


def find(tickets: list[dict], raw_id: str) -> dict:
    tid = normalize_id(raw_id)
    for t in tickets:
        if t["id"] == tid:
            return t
    sys.exit(f"error: no ticket {tid}")


def set_field(path: Path, key: str, value: str) -> None:
    text = path.read_text()
    m = FM_RE.match(text)
    fm = m.group(1)
    line = f"{key}: {value}"
    pattern = rf"^{re.escape(key)}:.*$"
    if re.search(pattern, fm, re.M):
        fm = re.sub(pattern, lambda _: line, fm, count=1, flags=re.M)
    else:
        fm += "\n" + line
    path.write_text(f"---\n{fm}\n---\n" + text[m.end():])


def format_value(key: str, value: str) -> str:
    if key == "status" and value not in STATUSES:
        sys.exit(f"error: status must be one of {STATUSES}")
    if key == "priority" and value not in PRIORITIES:
        sys.exit(f"error: priority must be one of {PRIORITIES}")
    if key == "estimate" and value not in ESTIMATES:
        sys.exit(f"error: estimate must be one of {ESTIMATES}")
    if key == "phase" and int(value) not in PHASES:
        sys.exit(f"error: phase must be one of {sorted(PHASES)}")
    if key == "id":
        sys.exit("error: rename the file and id by hand instead")
    if key in LIST_FIELDS:
        items = [x.strip() for x in value.strip("[]").split(",") if x.strip()]
        if key == "depends_on":
            items = [normalize_id(x) for x in items]
        return "[" + ", ".join(items) + "]"
    return value


def deps_closed(t: dict, by_id: dict) -> bool:
    return all(by_id.get(d, {}).get("status") in CLOSED for d in t.get("depends_on", []))


def row(t: dict) -> str:
    return (f"{t['id']}  {t['status']:<11}  {t.get('priority', ''):<3}  "
            f"ph{t['phase']}  {t.get('estimate', ''):<2}  {t.get('owner', ''):<12}  {t['title']}")


def cmd_list(args) -> None:
    for t in load_all():
        if args.status and t["status"] != args.status:
            continue
        if args.phase is not None and t["phase"] != args.phase:
            continue
        if args.owner and t.get("owner") != args.owner:
            continue
        if args.priority and t.get("priority") != args.priority:
            continue
        print(row(t))


def ready_tickets(tickets: list[dict]) -> list[dict]:
    by_id = {t["id"]: t for t in tickets}
    ready = [t for t in tickets if t["status"] == "todo" and deps_closed(t, by_id)]
    return sorted(ready, key=lambda t: (t.get("priority", "P9"), id_num(t["id"])))


def cmd_ready(args) -> None:
    for t in ready_tickets(load_all()):
        print(row(t))


def cmd_show(args) -> None:
    print(find(load_all(), args.id)["path"].read_text())


def cmd_set(args) -> None:
    if len(args.pairs) % 2:
        sys.exit("error: set takes KEY VALUE pairs")
    t = find(load_all(), args.id)
    for key, value in zip(args.pairs[::2], args.pairs[1::2], strict=False):
        set_field(t["path"], key, format_value(key, value))
    set_field(t["path"], "updated", dt.date.today().isoformat())
    print(f"updated {t['id']}")
    write_board()


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:50].rstrip("-")


def cmd_new(args) -> None:
    if args.phase not in PHASES:
        sys.exit(f"error: phase must be one of {sorted(PHASES)}")
    used = {id_num(t["id"]) for t in load_all()}
    start = 1 if args.phase == 0 else args.phase * 10
    free = [n for n in range(start, args.phase * 10 + 10) if n not in used]
    num = free[0] if free else max(used | {99}) + 1
    tid = f"T-{num:03d}"
    path = TICKET_DIR / f"{tid}-{slugify(args.title)}.md"
    path.write_text(TEMPLATE.read_text().replace("T-XXX", tid).replace("TITLE", args.title))
    fields = {
        "phase": str(args.phase),
        "priority": args.priority,
        "estimate": args.estimate,
        "depends_on": args.depends,
        "tags": args.tags,
        "status": "todo",
        "owner": "unassigned",
        "updated": dt.date.today().isoformat(),
    }
    for key, value in fields.items():
        set_field(path, key, format_value(key, value))
    print(f"created {path.relative_to(ROOT)}")
    write_board()


def check(tickets: list[dict]) -> list[str]:
    errors = []
    by_id = {}
    for t in tickets:
        name = t["path"].name
        if not name.startswith(t["id"] + "-"):
            errors.append(f"{name}: id {t['id']} does not match filename")
        if t["id"] in by_id:
            errors.append(f"{name}: duplicate id {t['id']}")
        by_id[t["id"]] = t
        if t["status"] not in STATUSES:
            errors.append(f"{t['id']}: bad status {t['status']!r}")
        if t.get("priority") not in PRIORITIES:
            errors.append(f"{t['id']}: bad priority {t.get('priority')!r}")
        if t.get("estimate") not in ESTIMATES:
            errors.append(f"{t['id']}: bad estimate {t.get('estimate')!r}")
        if t["phase"] not in PHASES:
            errors.append(f"{t['id']}: bad phase {t['phase']!r}")
    for t in tickets:
        for d in t.get("depends_on", []):
            if d not in by_id:
                errors.append(f"{t['id']}: depends on missing ticket {d}")
            elif t["status"] == "done" and by_id[d]["status"] not in CLOSED:
                errors.append(f"{t['id']}: done but dependency {d} is {by_id[d]['status']}")

    state: dict[str, int] = {}  # 1 = visiting, 2 = finished

    def visit(tid: str, trail: list[str]) -> None:
        if state.get(tid) == 2 or tid not in by_id:
            return
        if state.get(tid) == 1:
            errors.append("dependency cycle: " + " -> ".join(trail + [tid]))
            return
        state[tid] = 1
        for d in by_id[tid].get("depends_on", []):
            visit(d, trail + [tid])
        state[tid] = 2

    for tid in by_id:
        visit(tid, [])
    return errors


def cmd_check(args) -> None:
    tickets = load_all()
    errors = check(tickets)
    for e in errors:
        print("error:", e)
    if errors:
        sys.exit(1)
    print(f"ok: {len(tickets)} tickets")


def link(t: dict) -> str:
    return f"[{t['title']}]({t['path'].name})"


def write_board() -> None:
    tickets = load_all()
    counts = {s: sum(t["status"] == s for t in tickets) for s in STATUSES}
    closed = counts["done"] + counts["wontfix"]
    lines = [
        "# Ticket board",
        "",
        f"_Generated by `python3 scripts/tickets.py board` on {dt.date.today().isoformat()}. "
        "Do not edit by hand; edit the ticket files and regenerate._",
        "",
        f"**Progress:** {closed}/{len(tickets)} closed · {counts['in-progress']} in progress · "
        f"{counts['review']} in review · {counts['blocked']} blocked · {counts['todo']} todo",
        "",
        "## Ready to start",
        "",
        "Todo tickets whose dependencies are all closed, highest priority first.",
        "",
        "| ID | Title | Pri | Phase | Est | Owner |",
        "|---|---|---|---|---|---|",
    ]
    for t in ready_tickets(tickets):
        lines.append(f"| {t['id']} | {link(t)} | {t['priority']} | {t['phase']} | "
                     f"{t['estimate']} | {t['owner']} |")
    active = [t for t in tickets if t["status"] in {"in-progress", "review", "blocked"}]
    lines += ["", "## Active (in progress / review / blocked)", ""]
    if active:
        lines += ["| ID | Title | Status | Owner |", "|---|---|---|---|"]
        lines += [f"| {t['id']} | {link(t)} | {t['status']} | {t['owner']} |" for t in active]
    else:
        lines.append("_Nothing active._")
    for phase, name in PHASES.items():
        group = [t for t in tickets if t["phase"] == phase]
        if not group:
            continue
        done = sum(t["status"] in CLOSED for t in group)
        lines += [
            "",
            f"## Phase {phase} — {name} ({done}/{len(group)} closed)",
            "",
            "| ID | Title | Status | Pri | Est | Owner | Depends on |",
            "|---|---|---|---|---|---|---|",
        ]
        for t in group:
            deps = ", ".join(t.get("depends_on", [])) or "—"
            lines.append(f"| {t['id']} | {link(t)} | {t['status']} | {t['priority']} | "
                         f"{t['estimate']} | {t['owner']} | {deps} |")
    BOARD.write_text("\n".join(lines) + "\n")


def cmd_board(args) -> None:
    write_board()
    print(f"wrote {BOARD.relative_to(ROOT)}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list")
    s.add_argument("--status", choices=STATUSES)
    s.add_argument("--phase", type=int)
    s.add_argument("--owner")
    s.add_argument("--priority", choices=PRIORITIES)
    s.set_defaults(func=cmd_list)

    sub.add_parser("ready").set_defaults(func=cmd_ready)

    s = sub.add_parser("show")
    s.add_argument("id")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("set")
    s.add_argument("id")
    s.add_argument("pairs", nargs="+", metavar="KEY VALUE")
    s.set_defaults(func=cmd_set)

    s = sub.add_parser("new")
    s.add_argument("title")
    s.add_argument("--phase", type=int, required=True)
    s.add_argument("--priority", default="P1", choices=PRIORITIES)
    s.add_argument("--estimate", default="M", choices=ESTIMATES)
    s.add_argument("--depends", default="")
    s.add_argument("--tags", default="")
    s.set_defaults(func=cmd_new)

    sub.add_parser("board").set_defaults(func=cmd_board)
    sub.add_parser("check").set_defaults(func=cmd_check)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
