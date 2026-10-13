"""Command line: python -m aptevalkit {selftest,demo,checklist,summarize DIR [--deployed MODEL]}"""
import argparse
import glob
import json
import sys
import numpy as np
from .core import (bursts_from_gap, partition_groups, fpr_threshold, mcc, tie_report, saturation_report,
                   paired_summary, burst_bootstrap_recall_diff, print_checklist)
from .protocol import make_split, summarize


def selftest():
    rng = np.random.default_rng(0)
    ts = np.concatenate([np.arange(0, 10.0), 100 + np.arange(0, 10.0), 500 + np.arange(0, 20.0),
                         900 + np.arange(0, 10.0), 2000 + np.arange(0, 10.0)])
    assert bursts_from_gap(ts, 5.0).max() + 1 == 5 and bursts_from_gap(ts, 150.0).max() + 1 == 4
    ids = rng.integers(0, 60, 6000)
    sp = partition_groups(ids, rng)
    assert all(len(set(sp[ids == k])) == 1 for k in range(60)), "a group was split across partitions"
    share = np.bincount(sp) / len(sp)
    assert np.all(np.abs(share - [.6, .2, .2]) < .08), share
    b = rng.normal(size=1_000_000)
    fp = int((b >= fpr_threshold(b, 1e-3)).sum())
    assert 995 <= fp <= 1005, fp
    e = rng.exponential(size=100_000)
    c = np.minimum(e, np.quantile(e, .99))
    fp_clip = int((c >= fpr_threshold(c, 1e-4)).sum())
    assert fp_clip > 500, "tie trap not reproduced"
    print("tie trap: target FPR 0.01% (10 FP) gave", fp_clip, "FP on clipped scores;", tie_report(c, c[:10]))
    assert saturation_report(c)["share_within_tol_of_max"] > 0.009
    assert mcc(10, 0, 0, 10) == 1.0 and mcc(0, 5, 5, 0) == -1.0
    s = paired_summary(0.7 + rng.normal(0, .02, 20), 0.7 + rng.normal(0, .02, 20), n_train=60, n_test=20,
                       n_comparisons=8)
    assert {"mean", "ci_runs", "t_corrected", "p_bonferroni"} <= set(s), s
    m, ci = burst_bootstrap_recall_diff(rng.random(200) < .7, rng.random(200) < .7, rng.integers(0, 20, 200))
    assert ci[0] <= m <= ci[1]
    # split generators: no overlap, whole bursts never straddle partitions, chrono is ordered in time
    t = np.sort(rng.uniform(0, 1e5, 5000))
    y = (rng.random(5000) < 0.05).astype(int)
    for mode, benign in (("row", "random"), ("burst", "block"), ("burst", "random"), ("chrono", "block")):
        idx = make_split(t, y, mode, 5.0, benign, rng=np.random.default_rng(1))
        allidx = np.concatenate([idx["train"], idx["val"], idx["test"]])
        assert len(np.unique(allidx)) == len(allidx) == len(t), (mode, benign)
    idx = make_split(t, y, "burst", 5.0, "block", rng=np.random.default_rng(1))
    part = np.full(len(t), -1)
    for k, name in enumerate(("train", "val", "test")):
        part[idx[name]] = k
    a = np.where(y == 1)[0]
    bid = bursts_from_gap(t[a], 5.0)
    assert all(len(set(part[a][bid == g])) == 1 for g in np.unique(bid)), "a burst straddles partitions"
    idx = make_split(t, y, "chrono")
    assert t[idx["train"]].max() <= t[idx["val"]].min() and t[idx["val"]].max() <= t[idx["test"]].min()
    print("paired example:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k != "ci_runs"})
    print("selftest passed")


def _summarize_dir(path, deployed, exclude=()):
    files = sorted(glob.glob(path.rstrip("/") + "/*.json"))
    if not files:
        sys.exit(f"no .json result files in {path}")
    results = [json.load(open(f)) for f in files]
    if exclude:
        for r in results:
            r["res"] = {m: v for m, v in r["res"].items() if m not in exclude}
    rows, paired = summarize(results, deployed)
    print(f"{path}: {len(results)} runs")
    rules = list(dict.fromkeys(r["rule"] for r in rows))
    print(f"{'model':18s}" + "".join(f"{r:>24s}" for r in rules))
    for m in dict.fromkeys(r["model"] for r in rows):
        cells = []
        for k in rules:
            r = next((x for x in rows if x["model"] == m and x["rule"] == k), None)
            cells.append(f"{r['mcc_median']:>10.3f} [FP {r['fp_median']:>6.0f}]" if r else f"{'-':>22s}")
        print(f"{m:18s}" + "  ".join(cells))
    if paired:
        m = paired[0]['n_comparisons']
        print(f"\n{deployed} minus other: run-bootstrap CI; Nadeau-Bengio corrected t-test.")
        print(f"p_raw = unadjusted corrected-t p; p_adj = p_raw x {m} comparisons (Bonferroni, capped at 1). "
              f"Family = {m} models compared with {deployed}.")
        for p in paired:
            print(f"  {p['rule']:10s} {p['vs']:16s} {p['mean']:+.3f} [{p['ci_runs'][0]:+.3f}, {p['ci_runs'][1]:+.3f}] "
                  f"p_raw {p.get('p_corrected', float('nan')):.3f} p_adj {p.get('p_bonferroni', float('nan')):.3f} "
                  f"| {deployed} better {p['a_better']}/{p['runs']}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aptevalkit", description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("selftest")
    sub.add_parser("demo")
    sub.add_parser("checklist")
    s = sub.add_parser("summarize")
    s.add_argument("dir")
    s.add_argument("--deployed", default=None)
    s.add_argument("--exclude", default="", help="comma-separated models to leave out of the comparison family "
                   "(e.g. benign-only detectors that fail by a wide margin)")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "demo":
        from .demo import run_demo
        run_demo()
    elif a.cmd == "checklist":
        print_checklist({})
    elif a.cmd == "summarize":
        _summarize_dir(a.dir, a.deployed, tuple(x for x in a.exclude.split(",") if x))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
