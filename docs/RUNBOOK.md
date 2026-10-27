# Runbook: running the experiments

Everything below is runnable now. Each stage writes a `results/<run_id>/` directory with its
resolved `config.yaml` and `metadata.json` (git commit, package versions, device, timings).
Run the stages in order. A stage marked **decision** needs the student's sign-off before
moving on.

## 0. Environment

```bash
uv sync                      # Python >= 3.11; torch, transformers, jlens @ 581d398
uv run pytest -m "not slow"  # fast unit tests (no model weights needed)
uv run pytest                # all tests, incl. model/lens tests (downloads ~2 GB on first use)
```

**Device.** Run outside the Claude Code sandbox to use the Apple GPU (`device: auto` picks MPS,
with `dtype: auto` = bfloat16). Check speed first:

```bash
uv run python scripts/smoke_test.py --model Qwen/Qwen3.5-2B-Base --device mps --dtype bfloat16
```

On CPU, Qwen3.5 forward passes rely on the exact conv patch in
`src/icl_geometry/models/fast_kernels.py` (applied automatically). Expected CPU throughput is
~2 prompts/s for 2B-Base (`docs/COMPUTE.md`).

## 1. Items, tokenization, knowledge filter (Phase 1)

```bash
uv run python scripts/build_item_banks.py          # data/items/*.csv (already built)
uv run python scripts/validate_items.py            # results/item_bank_report.md
uv run python scripts/check_tokenization.py --model Qwen/Qwen3.5-2B-Base
uv run python scripts/run_knowledge.py --model Qwen/Qwen3.5-2B-Base   # T-013 (~25k short seqs)
```

**Decision (T-010/T-011):** domains (D-007) and an item spot-check.

## 2. Pilot (T-016, validation split only)

```bash
for p in exp2_pilot_arrow exp2_pilot_io exp1_pilot_arrow exp1_pilot_io; do
  uv run python scripts/make_prompts.py --config configs/prompts/$p.yaml
  uv run python scripts/run_behavior.py --config configs/runs/pilot_${p}_2b-base.yaml
done
uv run python scripts/analyze_pilot.py             # results/pilot/REPORT.md
```

**Decision (D-010):** template (`arrow` vs `io`), m grid, label sets, domains, main model.
Update `configs/prompts/exp{1,2}_main_*.yaml` to match.

## 3. Neutral representations (T-021)

```bash
uv run python scripts/run_neutral.py --model Qwen/Qwen3.5-2B-Base
uv run python scripts/neutral_sanity.py --model Qwen/Qwen3.5-2B-Base   # results/neutral_reps/
```

**Decision (D-011, D-012):** carrier averaging and outlier handling (re-check the top-1
variance share on 2B-Base).

Theory check on the real geometry (T-023; reads the cache only, ~30 s):

```bash
uv run python scripts/validate_theory_neutral.py --model Qwen/Qwen3.5-2B-Base
```

## 4. Experiment 1 (T-024, the MVP)

Lock the primary metrics first (D-005 → accepted).

```bash
for sp in val test; do
  uv run python scripts/make_prompts.py --config configs/prompts/exp1_main_$sp.yaml
  uv run python scripts/run_behavior.py --config configs/runs/exp1_main_${sp}_2b-base.yaml
done
uv run python scripts/run_exp1_analysis.py --model Qwen/Qwen3.5-2B-Base \
    --behavior results/<exp1 val run> results/<exp1 test run> \
    --select-split val --eval-split test
```

Exploratory variant on validation behavior only: `--select-split val --eval-split val` (the
report is labelled EXPLORATORY).

## 5. Experiment 2 (T-030, T-032, T-033)

```bash
for sp in val test; do
  uv run python scripts/make_prompts.py --config configs/prompts/exp2_main_$sp.yaml
  uv run python scripts/run_behavior.py --config configs/runs/exp2_main_${sp}_2b-base.yaml
  uv run python scripts/check_integrity.py --run results/<that run>
done
uv run python scripts/run_exp2_h2.py --run results/<exp2 test run>          # H2 (T-032)
uv run python scripts/run_exp2_h3.py --val-run results/<val> --test-run results/<test>  # H3
uv run python scripts/run_exp2_robustness.py --val-run results/<val> --test-run results/<test>  # T-034
```

The robustness table (`ROBUSTNESS.md`) classifies each claim × variation as holds / weakens /
fails / n/a (D-015). A second model size needs its own Exp 2 runs.

Activation caches: ~2 GB per split for 2B-Base (`data/cache/`, gitignored).

## 6. Experiment 3: Jacobian lens (T-041, T-042, T-043)

```bash
uv run python scripts/run_exp3_readout.py --run results/<exp2 test run>               # H4
uv run python scripts/run_exp3_readout.py --run results/<exp2 test run> --logit-lens  # baseline
uv run python scripts/run_exp3_jspace.py --val-run results/<val> --test-run results/<test>  # H5
```

`run_exp3_jspace.py` runs gradient pursuit (k = 25) over an 8k-word concept vocabulary at 3
middle layers by default. This is GPU-sized work; on CPU, use `--max-prompts` and
`--vocab-size` to subsample.

## 7. Figures (T-060, no model)

Fill in the run paths in `figures/manifest.yaml`, then:

```bash
uv run python scripts/make_figures.py     # figures/<name>.png + .pdf + .source.json
```

`make help` lists Makefile shortcuts for every stage.

## Smoke test of the whole pipeline (minutes, dev model)

```bash
for sp in val test; do
  uv run python scripts/make_prompts.py --config configs/prompts/smoke_exp2_$sp.yaml
  uv run python scripts/run_behavior.py --config configs/runs/smoke_exp2_$sp.yaml
done
V=results/<smoke val run>; T=results/<smoke test run>
uv run python scripts/check_integrity.py --run $V
uv run python scripts/run_exp2_h2.py --run $V --layer-step 4
uv run python scripts/run_exp2_h3.py --val-run $V --test-run $T --n-boot 100
uv run python scripts/run_exp3_readout.py --run $V --limit 40 --layer-step 4
uv run python scripts/run_exp3_jspace.py --val-run $V --test-run $T --max-prompts 150 \
    --vocab-size 2000 --n-random 2 --n-boot 50 --layers 10
```

Smoke outputs check the plumbing only. They are not results.

## Script index

| Stage | Script | Ticket |
|---|---|---|
| Model check | `scripts/smoke_test.py` | T-003 |
| Lens check | `scripts/jlens_sanity.py` | T-004 |
| Theory validation | `scripts/validate_theory_synthetic.py` | T-022/T-023 |
| Item banks | `scripts/build_item_banks.py`, `scripts/validate_items.py` | T-011 |
| Tokenization | `scripts/check_tokenization.py` | T-012 |
| Knowledge filter | `scripts/run_knowledge.py` | T-013 |
| Prompts | `scripts/make_prompts.py` | T-014 |
| Behavior + activations | `scripts/run_behavior.py`, `scripts/check_integrity.py` | T-015/T-020/T-030 |
| Pilot report | `scripts/analyze_pilot.py` | T-016 |
| Neutral reps | `scripts/run_neutral.py`, `scripts/neutral_sanity.py` | T-021 |
| Exp 1 | `scripts/run_exp1_analysis.py` | T-024 |
| Exp 2 | `scripts/run_exp2_h2.py`, `scripts/run_exp2_h3.py` | T-032/T-033 |
| Exp 3 | `scripts/run_exp3_readout.py`, `scripts/run_exp3_jspace.py` | T-041/T-042/T-043 |
| Theory on neutral reps | `scripts/validate_theory_neutral.py` | T-023 |
| Exp 2 robustness | `scripts/run_exp2_robustness.py` | T-034 |
| Report figures | `scripts/make_figures.py` (+ `figures/manifest.yaml`) | T-060 |
