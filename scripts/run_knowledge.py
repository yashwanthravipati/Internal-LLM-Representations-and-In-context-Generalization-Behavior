"""T-013: zero-shot knowledge check for every item bank.

    uv run python scripts/run_knowledge.py --model Qwen/Qwen3.5-2B-Base

Writes data/items/knowledge_<model>.parquet (validity table), the scored questions in
results/<run_id>/, and results/knowledge_check/REPORT_<model>.md. The question set itself
(data/prompts/knowledge_questions.jsonl) doubles as the explicit-question positive control for
the J-lens rule readout (T-041).
"""

from __future__ import annotations

import argparse

import pandas as pd

from icl_geometry import io
from icl_geometry.models.loading import load_model
from icl_geometry.tasks.knowledge import (
    exp1_questions,
    exp2_questions,
    score_questions,
    validity_table,
)
from icl_geometry.tasks.prompts import load_domains


def build_questions(domains: dict) -> list[dict]:
    rows = []
    cats = pd.read_csv(io.REPO_ROOT / domains["exp1"]["items"], dtype={"item": str})
    rows += exp1_questions(domains, cats)
    for name, spec in domains["exp2"].items():
        if name == "numbers":
            continue  # knowledge-free control; not filtered (T-013 notes)
        df = pd.read_csv(io.REPO_ROOT / spec["items"], dtype={"item": str})
        rows += exp2_questions(domains, name, df)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    domains = load_domains()
    rows = build_questions(domains)
    io.write_jsonl(rows, io.REPO_ROOT / "data" / "prompts" / "knowledge_questions.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    cfg = io.RunConfig(exp="t013", model=args.model, prompt_set="knowledge_questions",
                       desc="knowledge", batch_size=args.batch_size)
    with io.Run(cfg) as run:
        lm = load_model(args.model)
        scored = score_questions(lm, rows, args.batch_size, progress=True)
        io.write_parquet(scored, run.path("questions_scored.parquet"))
        valid = validity_table(scored)
        short = io.model_short_name(args.model)
        io.write_parquet(valid, io.REPO_ROOT / "data" / "items" / f"knowledge_{short}.parquet")

        lines = [f"# Knowledge check: {args.model}", "",
                 f"Run: `{run.dir.relative_to(io.REPO_ROOT)}`. Pass rule: strict (correct in "
                 "every phrasing/order/foil).", "",
                 "| Bank | Attribute | Items | Pass (strict) | Question accuracy |",
                 "|---|---|---|---|---|"]
        for (bank, attr), g in valid.groupby(["bank", "attribute"]):
            q = scored[(scored.bank == bank) & (scored.attribute == attr)]
            lines.append(f"| {bank} | {attr} | {len(g)} | {g.pass_strict.mean():.2f} | "
                         f"{q.correct.mean():.2f} |")
        # per category / cell pass counts
        cats = pd.read_csv(io.REPO_ROOT / domains["exp1"]["items"], dtype={"item": str})
        v1 = valid[valid.bank == "categories"].merge(cats[["item", "category"]], on="item")
        per_cat = v1.groupby("category").pass_strict.agg(["sum", "size"])
        lines += ["", "## Exp 1: items passing per category", "",
                  "| Category | Pass | Items |", "|---|---|---|"]
        lines += [f"| {c} | {int(r['sum'])} | {int(r['size'])} |" for c, r in per_cat.iterrows()]
        lines += ["", "## Exp 2: items passing both attributes, per cell", ""]
        for name, spec in domains["exp2"].items():
            if name == "numbers":
                continue
            attrs = list(spec["attributes"])
            items = pd.read_csv(io.REPO_ROOT / spec["items"], dtype={"item": str})
            ok = valid[valid.bank == name].groupby("item").pass_strict.all()
            items["pass"] = items.item.map(ok).fillna(False)
            cell = items.groupby(attrs)["pass"].agg(["sum", "size"])
            lines += [f"**{name}:** " + ", ".join(
                f"{a}/{b}: {int(r['sum'])}/{int(r['size'])}" for (a, b), r in cell.iterrows()),
                ""]
        report = io.REPO_ROOT / "results" / "knowledge_check" / f"REPORT_{short}.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(lines) + "\n")
        print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
