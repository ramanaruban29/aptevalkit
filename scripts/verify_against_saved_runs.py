"""Check that aptevalkit.make_split reproduces the splits of saved runs of the paper's runner.

Every saved result file records, per seed, the attack counts [train, val, test], the benign sizes and (for the
random / row / chrono modes) the stage mix of the test attacks. This script rebuilds each split from the raw
timestamps with make_split and compares all three. Run it from the folder that holds the runner's data files:

    python scripts/verify_against_saved_runs.py experiments_v4/cicapt_random5_block
    python scripts/verify_against_saved_runs.py experiments_v4/cicapt_row_block experiments_v4/cicapt_chrono_block

Needs numpy, pandas and the installed aptevalkit. Leave-one-stage-out folders are skipped (their draw order depends
on the stage loop of the runner).
"""
import argparse
import collections
import glob
import json
import sys
import numpy as np
import pandas as pd
import aptevalkit as ak

MODES = {"row": ("row", 5.0), "random5": ("burst", 5.0), "random120": ("burst", 120.0), "chrono": ("chrono", 5.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="result folders (one JSON per seed)")
    ap.add_argument("--ts", default="experiments_v4/ts_full.npy")
    ap.add_argument("--bursts", default="all_attack_bursts.csv")
    a = ap.parse_args()
    ts = np.load(a.ts)
    b = pd.read_csv(a.bursts)
    att_rows = b["csv_row"].values.astype(np.int64)
    stage_of = dict(zip(att_rows.tolist(), b["subLabel"].astype(str).str.lower().tolist()))
    y = np.zeros(len(ts), np.uint8)
    y[att_rows] = 1
    bad_total = 0
    for d in a.runs:
        files = sorted(glob.glob(d.rstrip("/") + "/*.json"))
        if not files:
            print(f"{d}: no result files"); continue
        n_ok = n_bad = 0
        for f in files:
            info = json.load(open(f))["info"]
            if info["mode"] not in MODES:
                print(f"{d}: skipping mode {info['mode']!r}"); break
            mode, gap = MODES[info["mode"]]
            idx = ak.make_split(ts, y, mode, gap, info.get("benign", "block"),
                                rng=np.random.default_rng(info["seed"]), attack_idx=att_rows)
            parts = [idx["train"], idx["val"], idx["test"]]
            attacks = [int(y[p].sum()) for p in parts]
            benign = [int((y[p] == 0).sum()) for p in parts]
            test_stages = dict(collections.Counter(stage_of[r] for r in idx["test"] if y[r] == 1))
            want_stages = {k.lower(): v for k, v in info.get("test_stages", {}).items()}
            problems = []
            if attacks != list(info["attacks"]):
                problems.append(f"attacks {attacks} != saved {info['attacks']}")
            if benign != list(info["benign_sizes"]):
                problems.append(f"benign {benign} != saved {info['benign_sizes']}")
            if want_stages and test_stages != want_stages:
                problems.append(f"test stages {test_stages} != saved {want_stages}")
            if problems:
                n_bad += 1
                print(f"  MISMATCH seed {info['seed']}: " + "; ".join(problems))
            else:
                n_ok += 1
        print(f"{d}: {n_ok} of {n_ok + n_bad} seeds reproduced exactly")
        bad_total += n_bad
    sys.exit(1 if bad_total else 0)


if __name__ == "__main__":
    main()
