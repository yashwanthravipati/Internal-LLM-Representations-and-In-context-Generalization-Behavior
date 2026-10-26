"""T-012: per-model token tables for items, labels and attribute words.

Writes data/items/token_info_<model>.parquet and results/tokenization/REPORT_<model>.md.

    uv run python scripts/check_tokenization.py --model Qwen/Qwen3.5-2B-Base
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import transformers
import yaml

from icl_geometry.io import model_short_name, write_parquet
from icl_geometry.tasks.templates import TEMPLATES
from icl_geometry.tasks.tokenization import item_tokens, single_token_id

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    domains = yaml.safe_load((ROOT / "configs" / "domains.yaml").read_text())
    short = model_short_name(args.model)

    banks = {"categories": ROOT / domains["exp1"]["items"]}
    banks |= {name: ROOT / spec["items"] for name, spec in domains["exp2"].items()}
    rows = []
    for bank, path in banks.items():
        df = pd.read_csv(path, dtype={"item": str})
        for r in df.itertuples():
            for tname in TEMPLATES:
                t = item_tokens(tok, r.item, tname)
                rows.append({"bank": bank, "item": r.item, "template": tname,
                             "n_tokens": t.n_tokens, "token_ids": t.token_ids,
                             "clean_boundary": t.clean_boundary, "suffix_ok": t.suffix_ok,
                             "confidence": r.confidence})
    info = pd.DataFrame(rows)
    out = ROOT / "data" / "items" / f"token_info_{short}.parquet"
    write_parquet(info, out)

    lines = [f"# Tokenization report: {args.model}", "",
             "Items are tokenized in context: rendered as the query of each template "
             "(`icl_geometry.tasks.templates`) after one demo line.", ""]

    # Labels
    lines += ["## Label sets", "", "| Set | Labels | Token ids | OK |", "|---|---|---|---|"]
    for name, labels in domains["label_sets"].items():
        ids = [single_token_id(tok, lbl.strip()) for lbl in labels]
        lines.append(f"| {name} | {labels} | {ids} | {all(i is not None for i in ids)} |")

    # Attribute words (with leading space) and meaningful labels
    lines += ["", "## Attribute words (leading space)", "",
              "| Domain | Attribute | Value | Word | Token id |", "|---|---|---|---|---|"]
    for dname, spec in domains["exp2"].items():
        for attr, aspec in spec["attributes"].items():
            for value, ws in aspec["words"].items():
                for w in ws:
                    lines.append(f"| {dname} | {attr} | {value} | {w} | "
                                 f"{single_token_id(tok, w)} |")
            ml = aspec.get("meaningful_labels", [])
            ml_ids = [single_token_id(tok, w.strip()) for w in ml]
            lines.append(f"| {dname} | {attr} | (meaningful labels) | {ml} | {ml_ids} |")

    # Items
    lines += ["", "## Items", "",
              "| Template | Bank | Items | 1 token | 2 tokens | 3+ tokens | Unclean boundary | "
              "Suffix changed |", "|---|---|---|---|---|---|---|---|"]
    for (tname, bank), g in info.groupby(["template", "bank"]):
        lines.append(
            f"| {tname} | {bank} | {len(g)} | {(g.n_tokens == 1).sum()} | "
            f"{(g.n_tokens == 2).sum()} | {(g.n_tokens >= 3).sum()} | "
            f"{(~g.clean_boundary).sum()} | {(~g.suffix_ok).sum()} |")
    bad = info[~info.clean_boundary | ~info.suffix_ok]
    if len(bad):
        lines += ["", "Items whose boundary or separator tokenization is not clean: "
                  + ", ".join(f"{r.template}/{r.bank}:{r.item}" for r in bad.head(30).itertuples())
                  + (f" ... ({len(bad)} total)" if len(bad) > 30 else ""), ""]
    report = ROOT / "results" / "tokenization" / f"REPORT_{short}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
