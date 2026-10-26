from collections import Counter

import pandas as pd
import pytest

from icl_geometry.tasks.prompts import PromptConfig, assign_splits, generate, load_domains


@pytest.fixture(scope="module")
def exp2_recs():
    cfg = PromptConfig(name="t2", experiment="exp2", split="test", m=[2, 4], n_demo_sets=2,
                       n_orders=2, conditions=["main", "random_label", "explicit_rule",
                                               "ambiguous"],
                       domains=["countries", "numbers", "words"])
    return cfg, generate(cfg)


@pytest.fixture(scope="module")
def exp1_recs():
    cfg = PromptConfig(name="t1", experiment="exp1", split="val", m=[1, 2], n_demo_sets=2,
                       n_orders=2, categories=["mammal", "bird", "fruit", "vehicle"],
                       conditions=["main", "random_label"])
    return cfg, generate(cfg)


def test_query_never_in_demos(exp1_recs, exp2_recs):
    for _, recs in (exp1_recs, exp2_recs):
        assert recs
        for r in recs:
            assert r["query_item"] not in {d["item"] for d in r["demos"]}
            s, e = r["query_span"]
            assert r["text"][s:e] == r["query_item"]


def test_exp2_demos_balanced_on_uncued(exp2_recs):
    _, recs = exp2_recs
    for r in recs:
        if r["condition"] not in ("main", "explicit_rule"):
            continue
        cued, uncued = r["cued_attr"], r["uncued_attr"]
        # each label has m demos, half from each uncued value
        by_label = Counter((d["label"], d[uncued]) for d in r["demos"])
        assert set(by_label.values()) == {r["m"] // 2}
        assert len(by_label) == 4
        # labels follow the cued attribute
        lab_of = {}
        for d in r["demos"]:
            assert lab_of.setdefault(d[cued], d["label"]) == d["label"]
        assert r["correct_label"] == lab_of[r["query_attrs"][cued]]
        assert r["incorrect_label"] != r["correct_label"]


def test_exp2_congruency_logic(exp2_recs):
    cfg, recs = exp2_recs
    domains = load_domains()
    for r in recs:
        spec = domains["exp2"][r["domain"]]["attributes"]
        cued, uncued = r["cued_attr"], r["uncued_attr"]
        ic = [str(v) for v in spec[cued]["values"]].index(r["query_attrs"][cued])
        iu = [str(v) for v in spec[uncued]["values"]].index(r["query_attrs"][uncued])
        assert r["congruent"] == (ic == iu)
    # queries are balanced 2 congruent : 2 incongruent per demo set
    c = Counter(r["congruent"] for r in recs if r["condition"] == "main")
    assert c[True] == c[False]


def test_random_label_breaks_both_rules(exp2_recs):
    _, recs = exp2_recs
    for r in recs:
        if r["condition"] != "random_label":
            continue
        for attr in (r["cued_attr"], r["uncued_attr"]):
            labels_per_value = {}
            for d in r["demos"]:
                labels_per_value.setdefault(d[attr], set()).add(d["label"])
            # no attribute value maps to a single label
            assert all(len(v) == 2 for v in labels_per_value.values())


def test_ambiguous_confounds_attributes(exp2_recs):
    _, recs = exp2_recs
    amb = [r for r in recs if r["condition"] == "ambiguous"]
    assert amb
    for r in amb:
        pairs = {(d[r["cued_attr"]], d[r["uncued_attr"]]) for d in r["demos"]}
        assert len(pairs) == 2  # only the two diagonal cells


def test_label_maps_counterbalanced(exp1_recs):
    _, recs = exp1_recs
    c = Counter((r["label_set"], r["label_map"], r["correct_label"]) for r in recs)
    assert len(c) == 4 * 2  # per label set: both labels correct under both mappings
    assert len(set(c.values())) == 1  # and equally often


def test_deterministic():
    cfg = PromptConfig(name="d", experiment="exp2", split="val", m=[2], n_demo_sets=1,
                       n_orders=1, domains=["numbers"])
    assert generate(cfg) == generate(cfg)


def test_splits_disjoint_and_stable():
    df = pd.DataFrame({"item": [f"x{i}" for i in range(40)], "cat": ["a"] * 20 + ["b"] * 20})
    s1 = assign_splits(df, ["cat"], 0.4)
    s2 = assign_splits(df.sample(frac=1, random_state=1), ["cat"], 0.4).sort_index()
    assert (s1 == s2).all()
    assert (s1[df.cat == "a"] == "val").sum() == 8


def test_val_test_items_disjoint():
    common = dict(experiment="exp2", m=[2], n_demo_sets=2, n_orders=1, domains=["words"])
    val = generate(PromptConfig(name="v", split="val", **common))
    test = generate(PromptConfig(name="t", split="test", **common))
    items = lambda recs: {d["item"] for r in recs for d in r["demos"]} | {  # noqa: E731
        r["query_item"] for r in recs}
    assert not items(val) & items(test)
