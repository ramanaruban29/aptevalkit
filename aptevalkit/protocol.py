"""Split generators and the repeated-run protocol runner (the 'leakage ladder')."""
import json
import os
import numpy as np
from .core import (bursts_from_gap, time_block_ids, partition_groups, fpr_threshold, metrics_at, mcc)

MODES = ("row", "burst", "chrono", "loso")
BENIGN = ("random", "block")


def make_split(ts, y, mode="burst", gap=5.0, benign="block", block_s=300.0, props=(0.6, 0.2, 0.2), rng=None,
               attack_idx=None, groups=None, held_out=None, val_share=0.25):
    """Return dict(train, val, test) of row indices into the full arrays.

    mode:   'row'    every attack flow is its own group (leaky reference: burst-mates sit on both sides);
            'burst'  whole attack bursts (consecutive gaps <= gap seconds) are assigned to partitions;
            'chrono' attack bursts ordered by start time and cut at the props boundaries (burst midpoint rule);
                     benign flows are cut at the start times of the first validation / test burst;
            'loso'   leave one group out (needs `groups`, one label per attack, and `held_out`): all attacks of
                     the held-out group form the test set, the rest is split train/val by bursts (val_share).
    benign: 'random' benign flows permuted and cut at the props boundaries;
            'block'  whole time blocks (block_s seconds, counted from the first flow in `ts`) are assigned.
            (Ignored for mode 'chrono'.)
    attack_idx: rows to treat as attacks, in the order their group ids / row groups are drawn (default: all y == 1,
                ascending). Group assignment for 'row' depends on this order.
    rng: numpy Generator. Draw order matches the paper's runner: attacks first, then benign flows ('loso': benign
         first, then attacks).
    """
    ts = np.asarray(ts, float)
    y = np.asarray(y).astype(int)
    rng = rng if rng is not None else np.random.default_rng(0)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if benign not in BENIGN:
        raise ValueError(f"benign must be one of {BENIGN}")
    a = np.asarray(attack_idx) if attack_idx is not None else np.where(y == 1)[0]
    b = np.where(y == 0)[0]
    part = np.full(len(ts), -1, dtype=int)
    ta = ts[a]

    def split_benign():
        if benign == "random":
            p = rng.permutation(b)
            c1, c2 = int(props[0] * len(p)), int((props[0] + props[1]) * len(p))
            part[p[:c1]] = 0
            part[p[c1:c2]] = 1
            part[p[c2:]] = 2
        else:
            blk = ((ts[b] - ts.min()) // block_s).astype(int)
            part[b] = partition_groups(blk, rng, props)

    if mode == "chrono":
        if len(a) == 0:
            raise ValueError("mode 'chrono' needs attack flows (y == 1)")
        ids = bursts_from_gap(ta, gap)
        nb = int(ids.max()) + 1
        if nb < 3:
            raise ValueError(f"mode 'chrono' needs at least 3 attack bursts at gap={gap}, found {nb}")
        sizes = np.bincount(ids, minlength=nb)
        start = np.full(nb, np.inf)
        np.minimum.at(start, ids, ta)
        order = np.argsort(start)
        mid = (np.cumsum(sizes[order]) - sizes[order] / 2) / sizes.sum()
        p = np.where(mid < props[0], 0, np.where(mid < props[0] + props[1], 1, 2))
        assign = np.empty(nb, dtype=int)
        assign[order] = p
        part[a] = assign[ids]
        c0, c1 = float(start[order][p == 1].min()), float(start[order][p == 2].min())
        tb = ts[b]
        part[b] = np.where(tb < c0, 0, np.where(tb < c1, 1, 2))
    elif mode == "loso":
        if groups is None or held_out is None:
            raise ValueError("mode 'loso' needs groups (one label per attack) and held_out")
        groups = np.asarray(groups)
        te = groups == held_out
        if not te.any():
            raise ValueError(f"held_out group {held_out!r} not found")
        split_benign()
        pa = np.full(len(a), 2, dtype=int)
        pa[~te] = partition_groups(bursts_from_gap(ta[~te], gap), rng, (1.0 - val_share, val_share))
        part[a] = pa
    else:
        ids = np.arange(len(a)) if mode == "row" else bursts_from_gap(ta, gap)
        part[a] = partition_groups(ids, rng, props)
        split_benign()
    return dict(train=np.where(part == 0)[0], val=np.where(part == 1)[0], test=np.where(part == 2)[0])


def _mcc_rule_threshold(val_scores, val_y):
    """Threshold maximising validation MCC over a grid of benign-score quantiles (uses validation labels only)."""
    vb = val_scores[val_y == 0]
    cands = np.unique(np.quantile(vb, 1.0 - np.logspace(-6, -0.5, 80)))
    best, best_m = cands[-1], -2.0
    va, vbn = val_scores[val_y == 1], vb
    for c in cands:
        m = metrics_at(c, vbn, va)["mcc"]
        if m > best_m:
            best, best_m = c, m
    return float(best)


def evaluate_scores(val_scores, val_y, test_scores, test_y, fpr_targets, mcc_rule=True):
    """Thresholds from validation benign scores (no test data), metrics on the test set."""
    val_scores, test_scores = np.asarray(val_scores, float), np.asarray(test_scores, float)
    val_y, test_y = np.asarray(val_y).astype(int), np.asarray(test_y).astype(int)
    tb, ta = test_scores[test_y == 0], test_scores[test_y == 1]
    out = {}
    for f in fpr_targets:
        thr = fpr_threshold(val_scores[val_y == 0], f)
        out[f"fpr{f:g}"] = metrics_at(thr, tb, ta)
    if mcc_rule:
        out["mccrule"] = metrics_at(_mcc_rule_threshold(val_scores, val_y), tb, ta)
    return out


def run_protocol(ts, y, fit_predict, mode="burst", gap=5.0, benign="block", seeds=range(10),
                 fpr_targets=(2e-5, 4e-5, 8e-5), mcc_rule=True, out_dir=None, props=(0.6, 0.2, 0.2), **split_kw):
    """Repeat the whole pipeline over seeds with redrawn splits.

    fit_predict(train_idx, val_idx, test_idx, seed) must (re)fit everything that sees training data (scaler,
    autoencoder, classifier) and return {model_name: (val_scores, test_scores)}; higher score = more anomalous.
    Extra keyword arguments (attack_idx, groups, held_out, ...) are passed to make_split.
    Returns a list of {'info': ..., 'res': {model: {rule: metrics}}}; also written as one JSON per seed if out_dir.
    """
    ts = np.asarray(ts, float)
    y = np.asarray(y).astype(int)
    results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = make_split(ts, y, mode, gap, benign, props=props, rng=rng, **split_kw)
        tr, va, te = idx["train"], idx["val"], idx["test"]
        scores = fit_predict(tr, va, te, seed)
        res = {m: evaluate_scores(v, y[va], t, y[te], fpr_targets, mcc_rule) for m, (v, t) in scores.items()}
        info = dict(mode=mode, benign=benign, gap=gap, seed=int(seed),
                    attacks=[int(y[i].sum()) for i in (tr, va, te)],
                    benign_sizes=[int((y[i] == 0).sum()) for i in (tr, va, te)])
        results.append(dict(info=info, res=res))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"seed{seed:02d}.json"), "w") as fh:
                json.dump(results[-1], fh)
    return results


def _rules(entry):
    """Metric entries of one model's result: keys whose value is a dict holding 'mcc' (skips e.g. stage_recall)."""
    return [k for k, v in entry.items() if isinstance(v, dict) and "mcc" in v]


def summarize(results, deployed=None):
    """Median MCC / recall / FP per model and rule, plus paired differences against `deployed` (if given).

    Tolerates result files in which some models lack some rules or carry extra non-metric entries.
    """
    from .core import paired_summary
    models = [m for m in results[0]["res"] if isinstance(results[0]["res"][m], dict)
              and all(m in r["res"] for r in results)]
    rows = []
    for m in models:
        for k in _rules(results[0]["res"][m]):
            if not all(k in r["res"][m] for r in results):
                continue
            v = [r["res"][m][k] for r in results]
            rows.append(dict(model=m, rule=k, runs=len(v), mcc_median=float(np.median([x["mcc"] for x in v])),
                             mcc_sd=float(np.std([x["mcc"] for x in v], ddof=1)) if len(v) > 1 else float("nan"),
                             recall_median=float(np.median([x["recall"] for x in v])),
                             fp_median=float(np.median([x["fp"] for x in v]))))
    paired = []
    if deployed and deployed in models:
        try:
            ntr = float(np.mean([r["info"]["attacks"][0] for r in results]))
            nte = float(np.mean([r["info"]["attacks"][2] for r in results]))
        except Exception:
            ntr, nte = 3.0, 1.0
        others = [m for m in models if m != deployed]
        for m in others:
            for k in _rules(results[0]["res"][deployed]):
                if not all(k in r["res"][m] and k in r["res"][deployed] for r in results):
                    continue
                a = [r["res"][deployed][k]["mcc"] for r in results]
                b = [r["res"][m][k]["mcc"] for r in results]
                paired.append(dict(rule=k, vs=m, **paired_summary(a, b, ntr, nte, n_comparisons=len(others))))
    return rows, paired
