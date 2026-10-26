"""Deterministic prompt generation for Exp 1 (category pairs) and Exp 2 (crossed domains).

Held-out discipline (D-009): every item gets a fixed `split` (val/test) by a stratified,
hash-based rule within its category (Exp 1) or crossed cell (Exp 2). A prompt set draws demos
and queries from a single split, so layer/hyperparameter selection (val) and headline numbers
(test) never share items. Within a prompt, the query is never one of the demos.

Exp 2 demos are balanced on the uncued attribute: each label gets m demos, half from each
value of the uncued attribute, so m must be even. Conditions:
  main          labels follow the cued attribute
  random_label  the same demo items, labels permuted so neither attribute predicts them
  explicit_rule main demos plus a stated rule before them
  ambiguous     cued and uncued attributes perfectly confounded in the demos (T-050)
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from icl_geometry.io import REPO_ROOT
from icl_geometry.tasks.templates import TEMPLATES

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
SPLIT_SALT = "icl-geometry-split-v1"


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def _item_hash(item: str) -> int:
    return int(hashlib.sha256(f"{SPLIT_SALT}:{item}".encode()).hexdigest()[:12], 16)


def assign_splits(df: pd.DataFrame, strata: list[str], val_frac: float = 0.4) -> pd.Series:
    """Stratified hash split: within each stratum, items sorted by a salted hash; the first
    round(val_frac * n) are 'val', the rest 'test'. Stable across runs and machines."""
    out = pd.Series("test", index=df.index, dtype=object)
    for _, g in df.groupby(strata):
        order = sorted(g.index, key=lambda i: _item_hash(str(df.at[i, "item"])))
        n_val = int(round(val_frac * len(order)))
        out.loc[order[:n_val]] = "val"
    return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PromptConfig:
    name: str
    experiment: str  # exp1 | exp2
    split: str  # val | test
    template: str = "arrow"
    seed: int = 0
    min_confidence: str = "high"
    val_frac: float = 0.4
    m: list[int] = field(default_factory=lambda: [2, 4, 8])
    label_sets: list[str] = field(default_factory=lambda: ["foo_bar", "zip_zap"])
    label_maps: list[int] = field(default_factory=lambda: [0, 1])
    n_demo_sets: int = 4
    n_orders: int = 3
    conditions: list[str] = field(default_factory=lambda: ["main"])
    # exp1
    categories: list[str] | None = None
    max_pairs: int | None = None
    n_queries_per_class: int = 1
    # exp2
    domains: list[str] | None = None
    min_confidence_by_domain: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PromptConfig:
        return cls(**yaml.safe_load(Path(path).read_text()))


def load_domains() -> dict[str, Any]:
    return yaml.safe_load((REPO_ROOT / "configs" / "domains.yaml").read_text())


def _filter_conf(df: pd.DataFrame, min_conf: str) -> pd.DataFrame:
    keep = df.confidence.map(CONFIDENCE_RANK) >= CONFIDENCE_RANK[min_conf]
    return df[keep]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _labels(domains: dict, label_set: str, label_map: int) -> tuple[str, str]:
    """(label for class/value 0, label for class/value 1); label_map=1 swaps them."""
    a, b = domains["label_sets"][label_set]
    return (a, b) if label_map == 0 else (b, a)


def _rng(seed: int, *keys: Any) -> np.random.Generator:
    h = hashlib.sha256(repr((seed,) + keys).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _orders(rng: np.random.Generator, n: int, n_orders: int) -> list[list[int]]:
    """n_orders distinct permutations of range(n) (fewer if n! < n_orders)."""
    perms: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    limit = 1
    for k in range(2, n + 1):
        limit *= k
    while len(perms) < min(n_orders, limit):
        p = tuple(int(x) for x in rng.permutation(n))
        if p not in seen:
            seen.add(p)
            perms.append(list(p))
    return perms


# ---------------------------------------------------------------------------
# Exp 1
# ---------------------------------------------------------------------------


def exp1_items(cfg: PromptConfig, domains: dict) -> pd.DataFrame:
    df = pd.read_csv(REPO_ROOT / domains["exp1"]["items"], dtype={"item": str})
    df = _filter_conf(df, cfg.min_confidence).copy()
    df["split"] = assign_splits(df, ["category"], cfg.val_frac)
    return df


def generate_exp1(cfg: PromptConfig, domains: dict) -> list[dict]:
    template = TEMPLATES[cfg.template]
    items = exp1_items(cfg, domains)
    items = items[items.split == cfg.split]
    by_cat = {c: list(g.item) for c, g in items.groupby("category")}
    sup = items.groupby("category").superordinate.first().to_dict()
    cats = sorted(cfg.categories or by_cat)
    pairs = list(itertools.combinations(cats, 2))
    if cfg.max_pairs is not None and len(pairs) > cfg.max_pairs:
        idx = _rng(cfg.seed, "pairs").choice(len(pairs), cfg.max_pairs, replace=False)
        pairs = [pairs[i] for i in sorted(idx)]
    records = []
    for (ca, cb), m, ls, lm in itertools.product(pairs, cfg.m, cfg.label_sets, cfg.label_maps):
        if min(len(by_cat[ca]), len(by_cat[cb])) <= m:
            continue  # not enough items for m demos plus a held-out query
        la, lb = _labels(domains, ls, lm)
        for ds in range(cfg.n_demo_sets):
            rng = _rng(cfg.seed, "exp1", ca, cb, m, ls, lm, ds)
            pa = [str(x) for x in rng.permutation(by_cat[ca])]
            pb = [str(x) for x in rng.permutation(by_cat[cb])]
            demos = [(x, la, ca) for x in pa[:m]] + [(x, lb, cb) for x in pb[:m]]
            queries = [(x, ca, la, lb) for x in pa[m : m + cfg.n_queries_per_class]] + [
                (x, cb, lb, la) for x in pb[m : m + cfg.n_queries_per_class]
            ]
            for oi, order in enumerate(_orders(rng, len(demos), cfg.n_orders)):
                d = [demos[i] for i in order]
                for q, qcat, correct, incorrect in queries:
                    for cond in cfg.conditions:
                        dd = d
                        if cond == "random_label":
                            dd = _permute_labels(d, rng)
                        elif cond != "main":
                            raise ValueError(f"exp1 condition {cond!r} not supported")
                        text, span = template.render([(x, lbl) for x, lbl, _ in dd], q)
                        records.append({
                            "experiment": "exp1", "domain": None,
                            "category_pair": [ca, cb], "superordinates": [sup[ca], sup[cb]],
                            "related": sup[ca] == sup[cb],
                            "cued_attr": None, "uncued_attr": None,
                            "m": m, "label_set": ls, "label_map": lm,
                            "demos": [{"item": x, "label": lbl, "category": c}
                                      for x, lbl, c in dd],
                            "query_item": q, "query_attrs": {"category": qcat},
                            "congruent": None, "condition": cond,
                            "correct_label": correct, "incorrect_label": incorrect,
                            "template_id": cfg.template, "demo_set": ds, "order": oi,
                            "seed": cfg.seed, "split": cfg.split, "text": text,
                            "query_span": list(span),
                        })
    return records


def _permute_labels(demos: list[tuple], rng: np.random.Generator) -> list[tuple]:
    """Shuffle labels across demos (label counts preserved)."""
    labels = [d[1] for d in demos]
    perm = rng.permutation(len(labels))
    return [(d[0], labels[p]) + tuple(d[2:]) for d, p in zip(demos, perm, strict=True)]


# ---------------------------------------------------------------------------
# Exp 2
# ---------------------------------------------------------------------------


def exp2_items(cfg: PromptConfig, domains: dict, domain: str) -> pd.DataFrame:
    spec = domains["exp2"][domain]
    attrs = list(spec["attributes"])
    df = pd.read_csv(REPO_ROOT / spec["items"], dtype={"item": str})
    for a in attrs:
        df[a] = df[a].astype(str)
    df = _filter_conf(df, cfg.min_confidence_by_domain.get(domain, cfg.min_confidence)).copy()
    df["split"] = assign_splits(df, attrs, cfg.val_frac)
    return df


def _random_label_demos(demos, attrs, rng, max_tries: int = 200):
    """Permute labels until neither attribute predicts them better than chance (each attribute
    value gets each label equally often). Falls back to the closest permutation found."""
    best, best_score = demos, None
    for _ in range(max_tries):
        cand = _permute_labels(demos, rng)
        score = 0
        for k in range(len(attrs)):
            for v in {d[2 + k] for d in cand}:
                labs = [d[1] for d in cand if d[2 + k] == v]
                score += abs(sum(lbl == cand[0][1] for lbl in labs) - len(labs) / 2)
        if best_score is None or score < best_score:
            best, best_score = cand, score
        if score == 0:
            break
    return best


def generate_exp2(cfg: PromptConfig, domains: dict) -> list[dict]:
    template = TEMPLATES[cfg.template]
    records = []
    for domain in cfg.domains or list(domains["exp2"]):
        spec = domains["exp2"][domain]
        attrs = list(spec["attributes"])
        df = exp2_items(cfg, domains, domain)
        df = df[df.split == cfg.split]
        values = {a: [str(v) for v in spec["attributes"][a]["values"]] for a in attrs}
        cells = {(r[attrs[0]], r[attrs[1]]): [] for _, r in df.iterrows()}
        for _, r in df.iterrows():
            cells[(r[attrs[0]], r[attrs[1]])].append(r["item"])
        for cued_i, m, ls, lm in itertools.product((0, 1), cfg.m, cfg.label_sets,
                                                   cfg.label_maps):
            if m % 2:
                raise ValueError("exp2 requires even m (demos balanced on the uncued attribute)")
            cued, uncued = attrs[cued_i], attrs[1 - cued_i]
            labs = _labels(domains, ls, lm)
            label_of = dict(zip(values[cued], labs, strict=True))
            for ds in range(cfg.n_demo_sets):
                rng = _rng(cfg.seed, "exp2", domain, cued, m, ls, lm, ds)
                pools = {c: [str(x) for x in rng.permutation(v)] for c, v in cells.items()}

                def cell(vc, vu, cued=cued, attrs=attrs):
                    return (vc, vu) if cued == attrs[0] else (vu, vc)

                # main demos: per cued value, m/2 from each uncued value
                demo_items, used = [], set()
                ok = True
                for vc in values[cued]:
                    for vu in values[uncued]:
                        pool = pools.get(cell(vc, vu), [])
                        if len(pool) < m // 2 + 1:
                            ok = False
                        demo_items += [(x, vc, vu) for x in pool[: m // 2]]
                        used.update(pool[: m // 2])
                if not ok:
                    continue  # cell too small for m demos plus a query
                # one query per cell (2 congruent, 2 incongruent)
                queries = []
                for vc in values[cued]:
                    for vu in values[uncued]:
                        rest = [x for x in pools[cell(vc, vu)] if x not in used]
                        queries.append((rest[0], vc, vu))
                base = [(x, label_of[vc], vc, vu) for x, vc, vu in demo_items]
                # ambiguous: m demos per label from the two "diagonal" cells only
                amb_demos = None
                if "ambiguous" in cfg.conditions:
                    amb_demos = []
                    query_items = {q[0] for q in queries}
                    for k, vc in enumerate(values[cued]):
                        vu = values[uncued][k]
                        pool = [x for x in pools[cell(vc, vu)] if x not in query_items]
                        if len(pool) >= m:
                            amb_demos += [(x, label_of[vc], vc, vu) for x in pool[:m]]
                        else:
                            amb_demos = None
                            break
                for oi, order in enumerate(_orders(rng, len(base), cfg.n_orders)):
                    for cond in cfg.conditions:
                        header = None
                        if cond == "main":
                            d = [base[i] for i in order]
                        elif cond == "random_label":
                            d = _random_label_demos([base[i] for i in order], [cued, uncued],
                                                    rng)
                        elif cond == "explicit_rule":
                            d = [base[i] for i in order]
                            v0, v1 = values[cued]
                            header = (f"Rule: label by {cued.replace('_', ' ')}. "
                                      f"{v0} ->{label_of[v0]}, {v1} ->{label_of[v1]}.\n\n")
                        elif cond == "ambiguous":
                            if amb_demos is None:
                                continue
                            d = [amb_demos[i] for i in _orders(rng, len(amb_demos), 1)[0]]
                        else:
                            raise ValueError(f"unknown condition {cond!r}")
                        for q, qc, qu in queries:
                            # congruent: both rules give the same label. Under the cued rule the
                            # label is label_of[qc]; under the uncued rule (value index aligned
                            # with label index) it is labs[values[uncued].index(qu)].
                            congruent = values[cued].index(qc) == values[uncued].index(qu)
                            text, span = template.render([(x, lbl) for x, lbl, *_ in d], q,
                                                         header=header)
                            correct = label_of[qc]
                            incorrect = labs[1 - labs.index(correct)]
                            records.append({
                                "experiment": "exp2", "domain": domain,
                                "category_pair": None, "superordinates": None, "related": None,
                                "cued_attr": cued, "uncued_attr": uncued,
                                "m": m, "label_set": ls, "label_map": lm,
                                "demos": [{"item": x, "label": lbl, cued: vc, uncued: vu}
                                          for x, lbl, vc, vu in d],
                                "query_item": q, "query_attrs": {cued: qc, uncued: qu},
                                "congruent": bool(congruent), "condition": cond,
                                "correct_label": correct, "incorrect_label": incorrect,
                                "template_id": cfg.template, "demo_set": ds, "order": oi,
                                "seed": cfg.seed, "split": cfg.split, "text": text,
                                "query_span": list(span),
                            })
    return records


def generate(cfg: PromptConfig) -> list[dict]:
    domains = load_domains()
    if cfg.experiment == "exp1":
        recs = generate_exp1(cfg, domains)
    elif cfg.experiment == "exp2":
        recs = generate_exp2(cfg, domains)
    else:
        raise ValueError(cfg.experiment)
    for i, r in enumerate(recs):
        r["prompt_id"] = f"{cfg.name}-{i:06d}"
    return recs
