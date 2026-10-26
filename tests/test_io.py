import json

import numpy as np
import pandas as pd
import pytest

from icl_geometry import io
from icl_geometry.utils.seed import seed_everything


def test_jsonl_roundtrip(tmp_path):
    recs = [{"prompt_id": "p0", "text": "dog -> foo"}, {"prompt_id": "p1", "demos": [1, 2]}]
    io.write_jsonl(recs, tmp_path / "x.jsonl")
    assert io.read_jsonl(tmp_path / "x.jsonl") == recs


def test_prompt_set_hash_stable_and_sensitive():
    a = [{"prompt_id": "p0", "text": "a", "m": 1}]
    b = [{"m": 1, "text": "a", "prompt_id": "p0"}]  # same content, different key order
    c = [{"prompt_id": "p0", "text": "b", "m": 1}]
    assert io.prompt_set_hash(a) == io.prompt_set_hash(b)
    assert io.prompt_set_hash(a) != io.prompt_set_hash(c)
    assert len(io.prompt_set_hash(a)) == 12


def test_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({"prompt_id": ["a", "b"], "margin": [0.5, -1.0], "correct": [True, False]})
    io.write_parquet(df, tmp_path / "b.parquet")
    pd.testing.assert_frame_equal(io.read_parquet(tmp_path / "b.parquet"), df)


def test_activation_cache_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    ids = [f"p{i}" for i in range(5)]
    vals = rng.standard_normal((5, 3, 8)).astype(np.float32)
    out = tmp_path / "cache" / "abc"
    w = io.ActivationWriter(out, ids, ["query_item", "decision"], n_states=3, d_model=8)
    w.write([0, 1, 2], "query_item", vals[:3])
    w.write([3, 4], "query_item", vals[3:])
    w.write(np.arange(5), "decision", vals)
    assert not out.exists()  # only the .partial dir exists until close()
    w.close()
    acts, index = io.load_activations(out, "query_item")
    assert acts.dtype == np.float16 and acts.shape == (5, 3, 8)
    np.testing.assert_allclose(acts, vals, atol=1e-2)
    assert list(index.prompt_id) == ids


def test_activation_writer_refuses_incomplete(tmp_path):
    w = io.ActivationWriter(tmp_path / "c", ["a", "b"], ["decision"], n_states=1, d_model=2)
    w.write([0], "decision", np.zeros((1, 1, 2)))
    with pytest.raises(RuntimeError):
        w.close()
    assert not (tmp_path / "c").exists()


def test_run_dir(tmp_path):
    cfg = io.RunConfig(exp="exp1", model="Qwen/Qwen3.5-0.8B", prompt_set="x", desc="unit")
    with io.Run(cfg, root=tmp_path) as run:
        run.path("out.txt").write_text("hi")
    meta = json.loads((run.dir / "metadata.json").read_text())
    assert meta["status"] == "ok" and "start" in meta and "end" in meta
    assert run.run_id.endswith("_exp1_qwen3.5-0.8b_unit")
    assert io.RunConfig.from_yaml(run.dir / "config.yaml").model == "Qwen/Qwen3.5-0.8B"


def test_seed_everything_reproducible():
    a = seed_everything(3).standard_normal(4)
    b = seed_everything(3).standard_normal(4)
    np.testing.assert_array_equal(a, b)
