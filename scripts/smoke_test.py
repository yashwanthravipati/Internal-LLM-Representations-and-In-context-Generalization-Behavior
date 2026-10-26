"""T-003 smoke test: load a model, run a 4-shot categorization prompt, check residual access,
right-padding invariance, and throughput.

    uv run python scripts/smoke_test.py --model Qwen/Qwen3.5-0.8B
"""

from __future__ import annotations

import argparse
import json
import resource
import time

import torch

from icl_geometry.models.loading import forward_with_states, load_model

PROMPT = "dog -> foo\noak -> bar\ncat -> foo\nfern -> bar\nhorse -> foo\nmaple -> bar\nwolf ->"
LABELS = (" foo", " bar")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-sizes", default="1,16,32")
    ap.add_argument("--prompt-tokens", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()
    lm = load_model(args.model, args.dtype, args.device)
    report: dict = {
        "model": args.model,
        "class": type(lm.model).__name__,
        "device": str(lm.device),
        "dtype": str(lm.dtype),
        "load_s": round(time.time() - t0, 1),
        "n_layers": lm.n_layers,
        "d_model": lm.d_model,
        "n_params_M": round(sum(p.numel() for p in lm.model.parameters()) / 1e6, 1),
        "layer_types": sorted(set(getattr(lm.text_config, "layer_types", None) or ["standard"])),
    }
    tok = lm.tokenizer
    label_ids = [tok.encode(lbl, add_special_tokens=False) for lbl in LABELS]
    report["label_token_ids"] = label_ids
    assert all(len(ids) == 1 for ids in label_ids), "labels must be single tokens"

    # 1. Few-shot prompt: top-5 next tokens and label probabilities.
    enc = tok(PROMPT, return_tensors="pt")
    logits, states = forward_with_states(lm, enc.input_ids)
    probs = logits[0, -1].float().softmax(-1)
    top = probs.topk(5)
    report["top5"] = [(tok.decode([int(i)]), round(float(p), 4)) for p, i in zip(*top, strict=True)]
    p_lab = [float(probs[ids[0]]) for ids in label_ids]
    report["p_labels"] = dict(zip(LABELS, [round(p, 4) for p in p_lab], strict=True))
    report["label_mass"] = round(sum(p_lab), 4)

    # 2. Residual access: n_layers + 1 states; HF's last hidden state = norm(last block output).
    report["states_shape"] = list(states.shape)
    with torch.no_grad():
        out = lm.model(input_ids=enc.input_ids.to(lm.device), output_hidden_states=True)
    hs = out.hidden_states
    report["hf_hidden_states_len"] = len(hs)
    last_block = states[0, :, -1]
    report["hf_last_equals_block_output"] = bool(torch.allclose(hs[-1][0], last_block, atol=1e-4))
    normed = lm.model.model.norm(last_block)
    report["hf_last_equals_norm_block_output"] = bool(torch.allclose(hs[-1][0], normed, atol=1e-4))
    mid = lm.n_layers // 2
    report["hf_mid_equals_block_output"] = bool(
        torch.allclose(hs[mid + 1][0], states[0, :, mid + 1], atol=1e-4)
    )
    # The lens readout of the final block output must reproduce the model's logits.
    report["unembed_reproduces_logits"] = bool(
        torch.allclose(lm.unembed(last_block)[-1].float(), logits[0, -1].float(), atol=1e-3)
    )

    # 3. Right padding does not change the states of the real tokens.
    short = "cat -> foo\noak ->"
    batch = tok([PROMPT, short], return_tensors="pt", padding=True)
    _, st_b = forward_with_states(lm, batch.input_ids, batch.attention_mask)
    n_short = int(batch.attention_mask[1].sum())
    _, st_s = forward_with_states(lm, tok(short, return_tensors="pt").input_ids)
    diff = (st_b[1, :n_short] - st_s[0]).abs().max().item()
    report["right_pad_max_abs_diff"] = diff

    # 4. Throughput on ~prompt_tokens-token prompts.
    filler = (PROMPT + "\n") * 10
    ids = tok(filler, return_tensors="pt").input_ids[:, : args.prompt_tokens]
    report["throughput"] = {}
    for bs in [int(b) for b in args.batch_sizes.split(",")]:
        x = ids.repeat(bs, 1)
        forward_with_states(lm, x)  # warm-up
        if lm.device.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        reps = 3
        for _ in range(reps):
            forward_with_states(lm, x)
        if lm.device.type == "cuda":
            torch.cuda.synchronize()
        report["throughput"][bs] = round(reps * bs / (time.time() - t), 2)
    if lm.device.type == "cuda":
        report["peak_mem_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    else:
        # ru_maxrss is bytes on macOS, KiB on Linux
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["peak_rss_GB"] = round(rss / (1e9 if rss > 1e8 else 1e6), 2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
