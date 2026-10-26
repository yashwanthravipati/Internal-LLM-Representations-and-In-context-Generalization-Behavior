"""T-021: neutral representations for every item in every bank.

    uv run python scripts/run_neutral.py --model Qwen/Qwen3.5-2B-Base

Cache: data/cache/<model>/neutral_<hash>/states.npy, float16
[n_items, n_carriers, n_layers + 1, d_model], plus index.parquet (row -> item). The hash covers
the item list and the carrier texts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from icl_geometry import io
from icl_geometry.models.loading import load_model
from icl_geometry.models.neutral import CARRIERS, neutral_states
from icl_geometry.tasks.prompts import load_domains


def all_items() -> list[str]:
    domains = load_domains()
    paths = [domains["exp1"]["items"]] + [s["items"] for s in domains["exp2"].values()]
    items: list[str] = []
    for p in paths:
        items += list(pd.read_csv(io.REPO_ROOT / p, dtype={"item": str}).item)
    return sorted(set(items))


def neutral_cache_dir(model: str, items: list[str]) -> os.PathLike:
    h = hashlib.sha256(json.dumps([items, list(CARRIERS)]).encode()).hexdigest()[:12]
    return io.activation_dir(model, f"neutral_{h}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    items = all_items()
    out = neutral_cache_dir(args.model, items)
    if out.exists():
        print("exists:", out)
        return
    t0 = time.time()
    lm = load_model(args.model)
    arr = neutral_states(lm, items, args.batch_size, progress=True)
    tmp = out.with_name(out.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / "states.npy", arr)
    io.write_parquet(pd.DataFrame({"row": range(len(items)), "item": items}),
                     tmp / "index.parquet")
    (tmp / "meta.json").write_text(json.dumps({
        "model": args.model, "carriers": list(CARRIERS), "shape": list(arr.shape),
        "seconds": round(time.time() - t0, 1), "git": io.git_state(),
    }, indent=2))
    os.replace(tmp, out)
    print(f"{arr.shape} -> {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
