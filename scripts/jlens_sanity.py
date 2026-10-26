"""T-004: load a pre-fitted Jacobian lens, inspect its metadata, reproduce a paper-style latent
readout, check our residual convention against jlens, and probe a toy few-shot prompt.

    uv run python scripts/jlens_sanity.py --model Qwen/Qwen3.5-0.8B \
        --lens qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt
"""

from __future__ import annotations

import argparse
import json

import jlens
import torch
from huggingface_hub import hf_hub_download

from icl_geometry import io
from icl_geometry.models.loading import forward_with_states, load_model

LENS_REPO = "neuronpedia/jacobian-lens"
PROMPTS = {
    # Paper example: the latent "Italy" should appear in middle layers before "euro".
    "boot_currency": ("Fact: The currency used in the country shaped like a boot is", [-2, -1]),
    # Toy categorization: does the category ("animal"/"plant"/"tree") show up?
    "fewshot_animal_plant": (
        "dog -> foo\noak -> bar\ncat -> foo\nfern -> bar\nhorse -> foo\nmaple -> bar\nwolf ->",
        [-2, -1],  # -2 = query item " wolf", -1 = decision position " ->"
    ),
}


def topk_tokens(tok, logits: torch.Tensor, k: int = 8) -> list[str]:
    return [tok.decode([int(i)]) for i in logits.topk(k).indices]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True, help="path inside neuronpedia/jacobian-lens")
    ap.add_argument("--desc", default="jlens-sanity")
    args = ap.parse_args()

    cfg = io.RunConfig(exp="t004", model=args.model, prompt_set="inline", desc=args.desc,
                       extra={"lens_repo": LENS_REPO, "lens_file": args.lens})
    with io.Run(cfg) as run:
        path = hf_hub_download(LENS_REPO, args.lens)
        raw = torch.load(path, map_location="cpu", weights_only=True)
        meta = {k: v for k, v in raw.items() if k != "J"}
        meta["J_layers"] = sorted(raw["J"])
        meta["J_dtype"] = str(next(iter(raw["J"].values())).dtype)
        lens = jlens.JacobianLens.load(path)

        lm = load_model(args.model)
        model = jlens.from_hf(lm.model, lm.tokenizer)
        tok = lm.tokenizer
        report: dict = {"lens_meta": meta, "lens": repr(lens), "model": repr(model)}

        # Convention check: jlens layer l == our state index l + 1.
        ids = model.encode(PROMPTS["boot_currency"][0])
        _, states = forward_with_states(lm, ids)
        with jlens.ActivationRecorder(model.layers, at=[0, 5, model.n_layers - 1]) as rec:
            model.forward(ids)
        report["convention_max_abs_diff"] = {
            str(layer): float((rec.activations[layer][0] - states[0, :, layer + 1]).abs().max())
            for layer in rec.activations
        }

        readouts = {}
        for name, (prompt, positions) in PROMPTS.items():
            toks = [tok.decode([int(t)]) for t in model.encode(prompt)[0]]
            for use_j in (True, False):
                lens_logits, model_logits, _ = lens.apply(
                    model, prompt, positions=positions, use_jacobian=use_j
                )
                key = f"{name}/{'jlens' if use_j else 'logit_lens'}"
                readouts[key] = {
                    "positions": {str(p): toks[p] for p in positions},
                    "model_top": {str(p): topk_tokens(tok, model_logits[i])
                                  for i, p in enumerate(positions)},
                    "layers": {
                        str(layer): {str(p): topk_tokens(tok, lg[i]) for i, p in
                                     enumerate(positions)}
                        for layer, lg in sorted(lens_logits.items())
                    },
                }
        report["readouts"] = readouts
        run.path("report.json").write_text(json.dumps(report, indent=1))

        # Compact console view
        print(json.dumps({k: report[k] for k in ("lens_meta", "convention_max_abs_diff")},
                         indent=1, default=str))
        for key, r in readouts.items():
            print(f"\n=== {key}  positions={r['positions']}  model_top={r['model_top']}")
            for layer, per_pos in r["layers"].items():
                print(f"  L{int(layer):2d} " + " | ".join(
                    f"{p}: {' '.join(repr(t) for t in v[:6])}" for p, v in per_pos.items()))
        print("run dir:", run.dir)


if __name__ == "__main__":
    main()
