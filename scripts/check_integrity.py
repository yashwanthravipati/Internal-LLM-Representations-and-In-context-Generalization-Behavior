"""T-030: data-integrity check for a scored run with cached activations.

Checks that every activation row <-> prompt_id <-> behavior row matches, that there are no
NaNs/infs in the cache, and prints the per-cell prompt counts against the design.

    uv run python scripts/check_integrity.py --run results/<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from icl_geometry import io


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = io.REPO_ROOT / args.run
    cfg = io.RunConfig.from_yaml(run / "config.yaml")
    meta = json.loads((run / "metadata.json").read_text())
    prompts = pd.DataFrame(io.read_jsonl(io.REPO_ROOT / cfg.prompt_set))
    if cfg.extra.get("limit"):
        prompts = prompts.iloc[: cfg.extra["limit"]]
    beh = pd.read_parquet(run / "behavior.parquet")
    errors = []
    if meta.get("prompt_set_hash") != io.prompt_set_hash(prompts.to_dict("records")):
        errors.append("prompt set hash differs from the one recorded in metadata.json")
    if set(beh.prompt_id) != set(prompts.prompt_id) or beh.prompt_id.duplicated().any():
        errors.append("behavior rows don't match prompt ids one-to-one")
    if beh[["margin", "label_mass"]].isna().any().any():
        errors.append("NaN in behavior")
    if "activation_cache" in meta:
        cache = io.REPO_ROOT / meta["activation_cache"]
        for pos in cfg.positions:
            acts, index = io.load_activations(cache, pos)
            if list(index.prompt_id) != list(beh.prompt_id):
                errors.append(f"{pos}: activation index order != behavior order")
            if acts.shape[0] != len(beh):
                errors.append(f"{pos}: {acts.shape[0]} activation rows vs {len(beh)} prompts")
            sample = np.asarray(acts[:: max(1, len(acts) // 200)], dtype=np.float32)
            if not np.isfinite(sample).all():
                errors.append(f"{pos}: non-finite values in activations")
    cols = [c for c in ("experiment", "domain", "cued_attr", "condition", "m", "congruent")
            if c in prompts and prompts[c].notna().any()]
    print(prompts.groupby(cols, dropna=False).size().to_string())
    for e in errors:
        print("FAIL:", e)
    print("OK" if not errors else f"{len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
