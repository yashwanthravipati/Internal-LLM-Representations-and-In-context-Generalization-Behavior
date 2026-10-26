import numpy as np
import pytest

from icl_geometry.models.neutral import CARRIERS, carrier_texts
from icl_geometry.tasks.tokenization import span_token_indices

MODEL = "Qwen/Qwen3.5-0.8B"


def test_carrier_spans():
    for text, (s, e) in carrier_texts("sea lion"):
        assert text[s:e] == "sea lion" and s > 0


@pytest.fixture(scope="module")
def lm():
    from icl_geometry.models.loading import load_model

    try:
        return load_model(MODEL, "float32", "cpu")
    except OSError:
        pytest.skip("model weights not cached")


@pytest.mark.slow
def test_positions_decode_to_item_and_shapes(lm):
    from icl_geometry.models.activations import extract_states

    tok = lm.tokenizer
    texts, spans = zip(*carrier_texts("hedgehog"), strict=True)
    states, qidx = extract_states(lm, list(texts), list(spans), ("query_item", "decision"))
    for text, span, qi in zip(texts, spans, qidx, strict=True):
        inside, ids, _ = span_token_indices(tok, text, span)
        assert qi == inside[-1]
        # the recorded token is the item's last subtoken
        assert "hedgehog".endswith(tok.decode([ids[qi]]).strip())
    assert states["query_item"].shape == (len(CARRIERS), lm.n_layers + 1, lm.d_model)


@pytest.mark.slow
def test_alone_vs_batch(lm):
    from icl_geometry.models.neutral import neutral_states

    batch = neutral_states(lm, ["dog", "chrysanthemum", "oak"], batch_size=12)
    alone = neutral_states(lm, ["chrysanthemum"], batch_size=1)
    np.testing.assert_allclose(batch[1].astype(np.float32), alone[0].astype(np.float32),
                               atol=2e-2, rtol=1e-2)
