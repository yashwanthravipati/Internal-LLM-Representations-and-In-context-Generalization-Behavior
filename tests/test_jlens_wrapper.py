import pytest
import torch

MODEL = "Qwen/Qwen3.5-0.8B"
PROMPT = "dog -> foo\noak -> bar\ncat -> foo\nfern -> bar\nwolf ->"


@pytest.fixture(scope="module")
def setup():
    from icl_geometry.lens.jlens_wrapper import Lens
    from icl_geometry.models.loading import load_model

    try:
        lm = load_model(MODEL, "float32", "cpu")
        lens = Lens.load(MODEL)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"model or lens not cached: {e}")
    return lm, lens


@pytest.mark.slow
def test_readout_matches_jlens_apply(setup):
    import jlens

    from icl_geometry.lens.jlens_wrapper import readout, state_index
    from icl_geometry.models.loading import forward_with_states

    lm, lens = setup
    ref_lens = jlens.JacobianLens({k: v for k, v in lens.J.items()}, n_prompts=lens.n_prompts,
                                  d_model=lm.d_model)
    ref_model = jlens.from_hf(lm.model, lm.tokenizer)
    layers = [3, 12, 20]
    ref, _, ids = ref_lens.apply(ref_model, PROMPT, layers=layers, positions=[-2, -1])
    _, states = forward_with_states(lm, ids)
    for layer in layers:
        ours = readout(lm, lens, states[0, [-2, -1], state_index(layer)], layer)
        torch.testing.assert_close(ours, ref[layer], atol=1e-3, rtol=1e-3)


@pytest.mark.slow
def test_concept_vector_ranking_matches_readout(setup):
    from icl_geometry.lens.jlens_wrapper import (
        concept_scores,
        concept_vectors,
        readout,
        state_index,
    )
    from icl_geometry.models.loading import forward_with_states

    lm, lens = setup
    ids = lm.tokenizer(PROMPT, return_tensors="pt").input_ids
    _, states = forward_with_states(lm, ids)
    layer = 15
    h = states[0, -1:, state_index(layer)]
    logits = readout(lm, lens, h, layer)[0]
    tokens = logits.topk(50).indices.tolist() + [100, 2000, 30000]
    vecs = concept_vectors(lm, lens, tokens, layer)
    assert vecs.shape == (len(tokens), lm.d_model)
    dots = concept_scores(h, vecs)["dot"][0]
    # same ordering as the lens logits for these tokens
    assert torch.equal(dots.argsort(), logits[tokens].argsort())


@pytest.mark.slow
def test_percentile_ranks(setup):
    import numpy as np

    from icl_geometry.lens.jlens_wrapper import readout, state_index
    from icl_geometry.lens.readout import percentile_ranks
    from icl_geometry.models.loading import forward_with_states

    lm, lens = setup
    ids = lm.tokenizer(PROMPT, return_tensors="pt").input_ids
    _, states = forward_with_states(lm, ids)
    st = states[0, -2:].numpy()  # [2, n_states, d]
    logits = readout(lm, lens, states[0, -1:, state_index(15)], 15)[0]
    top = int(logits.argmax())
    ranks = percentile_ranks(lm, lens, st, [15], [top, 100])
    assert ranks.shape == (2, 1, 2)
    assert ranks[1, 0, 0] == 1.0  # the top token is at the top of the vocabulary
    expected = float((logits < logits[100]).sum()) / (logits.numel() - 1)
    np.testing.assert_allclose(ranks[1, 0, 1], expected, atol=1e-6)
