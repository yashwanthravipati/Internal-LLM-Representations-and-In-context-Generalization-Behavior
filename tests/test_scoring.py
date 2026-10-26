import numpy as np
import pytest
import torch

from icl_geometry.behavior.scoring import content_free_text, score_records
from icl_geometry.tasks.prompts import PromptConfig, generate

MODEL = "Qwen/Qwen3.5-0.8B"


@pytest.fixture(scope="module")
def lm():
    from icl_geometry.models.loading import load_model

    try:
        return load_model(MODEL, "float32", "cpu")
    except OSError:
        pytest.skip("model weights not cached")


@pytest.fixture(scope="module")
def records():
    cfg = PromptConfig(name="s", experiment="exp2", split="val", m=[2, 4], n_demo_sets=1,
                       n_orders=1, domains=["numbers", "words"],
                       conditions=["main", "explicit_rule"])
    return generate(cfg)[:12]


def test_content_free_text(records):
    r = records[0]
    cf = content_free_text(r)
    assert cf.endswith("N/A ->") and cf.startswith(r["text"][: r["query_span"][0]])


@pytest.mark.slow
def test_padding_invariance_and_positions(lm, records):
    """Scores and states for a prompt scored alone equal those inside a padded batch."""
    df_b, st_b = score_records(lm, records, batch_size=6, calibrate=False,
                               positions=("query_item", "decision"))
    for i in (0, 5, 11):
        df_1, st_1 = score_records(lm, [records[i]], batch_size=1, calibrate=False,
                                   positions=("query_item", "decision"))
        np.testing.assert_allclose(df_b.margin[i], df_1.margin[0], atol=1e-3)
        for p in st_b:
            np.testing.assert_allclose(st_b[p][i].astype(np.float32),
                                       st_1[p][0].astype(np.float32), atol=2e-2, rtol=1e-2)
    assert ((df_b.label_mass > 0) & (df_b.label_mass <= 1)).all()


@pytest.mark.slow
def test_label_logps_match_direct_forward(lm, records):
    r = records[0]
    df, _ = score_records(lm, [r], batch_size=1, calibrate=False)
    tok = lm.tokenizer
    ids = tok(r["text"], return_tensors="pt").input_ids
    with torch.no_grad():
        lp = lm.model(ids).logits[0, -1].log_softmax(-1)
    a = tok.encode(r["correct_label"], add_special_tokens=False)[0]
    np.testing.assert_allclose(df.logp_correct[0], lp[a].item(), atol=1e-4)
