"""Score a prompt set (T-015) and optionally cache residual states (T-020).

    uv run python scripts/run_behavior.py --config configs/runs/pilot_exp2_0.8b.yaml

Config keys (icl_geometry.io.RunConfig): model, prompt_set (JSONL path), positions, batch_size,
dtype, device, seed, plus extras: `calibrate` (default true), `cache_activations` (default
false), `limit` (first N prompts, for smoke runs).

Outputs: results/<run_id>/behavior.parquet, summary.json; activations (if requested) in
data/cache/<model>/<prompt_set_hash>/ with the hash recorded in metadata.json.
"""

from __future__ import annotations

import argparse
import json
import time

from icl_geometry import io
from icl_geometry.behavior.scoring import score_records
from icl_geometry.models.loading import load_model
from icl_geometry.utils.seed import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = io.RunConfig.from_yaml(args.config)
    seed_everything(cfg.seed)
    records = io.read_jsonl(io.REPO_ROOT / cfg.prompt_set)
    if cfg.extra.get("limit"):
        records = records[: cfg.extra["limit"]]
    set_hash = io.prompt_set_hash(records)
    cache = cfg.extra.get("cache_activations", False)
    positions = tuple(cfg.positions) if cache else ()

    with io.Run(cfg) as run:
        run.metadata["prompt_set_hash"] = set_hash
        run.metadata["n_prompts"] = len(records)
        t0 = time.time()
        lm = load_model(cfg.model, cfg.dtype, cfg.device)
        df, states = score_records(lm, records, cfg.batch_size,
                                   calibrate=cfg.extra.get("calibrate", True),
                                   positions=positions, progress=True)
        run.metadata["forward_seconds"] = round(time.time() - t0, 1)
        io.write_parquet(df, run.path("behavior.parquet"))
        if cache:
            out = io.activation_dir(cfg.model, set_hash)
            if out.exists():
                print("activation cache exists, not overwriting:", out)
            else:
                w = io.ActivationWriter(out, list(df.prompt_id), list(positions),
                                        lm.n_layers + 1, lm.d_model)
                for p in positions:
                    w.write(list(range(len(df))), p, states[p])
                w.close()
            run.metadata["activation_cache"] = str(out.relative_to(io.REPO_ROOT))
        summary = {
            "n": len(df),
            "accuracy": float(df.correct.mean()),
            "median_label_mass": float(df.label_mass.median()),
        }
        if "correct_calibrated" in df:
            summary["accuracy_calibrated"] = float(df.correct_calibrated.mean())
        run.path("summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), "\nrun dir:", run.dir)


if __name__ == "__main__":
    main()
