"""Generate a prompt set from a config (T-014).

    uv run python scripts/make_prompts.py --config configs/prompts/exp2_pilot.yaml

Writes data/prompts/<name>.jsonl and prints a summary.
"""

from __future__ import annotations

import argparse
from collections import Counter

from icl_geometry import io
from icl_geometry.tasks.prompts import PromptConfig, generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = PromptConfig.from_yaml(args.config)
    recs = generate(cfg)
    out = io.REPO_ROOT / "data" / "prompts" / f"{cfg.name}.jsonl"
    io.write_jsonl(recs, out)
    print(f"{len(recs)} prompts -> {out} (hash {io.prompt_set_hash(recs)})")
    keys = ["domain", "condition", "m", "congruent"] if cfg.experiment == "exp2" else [
        "related", "m", "condition"]
    for k in keys:
        print(f"  {k}: {dict(sorted(Counter(str(r[k]) for r in recs).items()))}")
    if recs:
        print("\nExample:\n" + recs[0]["text"])


if __name__ == "__main__":
    main()
