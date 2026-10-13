"""Synthetic demonstration of the leakage ladder (runs in seconds, needs only numpy).

Attacks come in bursts of near-duplicate flows. A memory-based detector ('nn') looks a flow up among training attacks,
so it is rewarded by any split that puts burst-mates on both sides. A linear detector ('lda') learns only the shared
attack signal and is unaffected. The ladder therefore separates leakage from real detection ability.
"""
import numpy as np
from .protocol import run_protocol, summarize


def make_synthetic(seed=0, n_benign=20000, n_bursts=40, span=36000.0):
    rng = np.random.default_rng(seed)
    d = 6
    tb = np.sort(rng.uniform(0, span, n_benign))
    xb = rng.normal(size=(n_benign, d)) + 0.5 * np.sin(tb / 3000.0)[:, None]      # slow benign drift
    ts, xs = [tb], [xb]
    starts = np.sort(rng.uniform(0, span, n_bursts))
    for s in starts:
        k = int(rng.integers(8, 28))
        centre = rng.normal(size=d)                                                # burst-specific signature
        centre[:2] += 3.0                                                          # shared attack signal
        ts.append(s + 0.5 * np.arange(k))
        xs.append(centre + 0.15 * rng.normal(size=(k, d)))
    ts = np.concatenate(ts)
    X = np.vstack(xs)
    y = np.concatenate([np.zeros(n_benign, int), np.ones(len(ts) - n_benign, int)])
    return ts, X, y


def _nn_dist(Q, R, chunk=2000):
    out = np.empty(len(Q))
    r2 = (R ** 2).sum(1)
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk]
        d2 = (q ** 2).sum(1)[:, None] + r2[None, :] - 2 * q @ R.T
        out[i:i + chunk] = np.sqrt(np.maximum(d2.min(1), 0))
    return out


def make_fit_predict(X, y):
    def fit_predict(tr, va, te, seed):
        rng = np.random.default_rng(1000 + seed)
        ben = tr[y[tr] == 0]
        att = tr[y[tr] == 1]
        mu_b = X[ben].mean(0)
        sd_b = X[ben].std(0) + 1e-9
        Z = (X - mu_b) / sd_b
        ref = Z[rng.choice(ben, min(2000, len(ben)), replace=False)]
        direction = Z[att].mean(0) - Z[ben].mean(0)
        scores = {}
        for m in ("nn", "lda"):
            pair = []
            for idx in (va, te):
                if m == "nn":
                    s = _nn_dist(Z[idx], ref) - _nn_dist(Z[idx], Z[att])
                else:
                    s = Z[idx] @ direction
                pair.append(s)
            scores[m] = tuple(pair)
        return scores
    return fit_predict


def run_demo(seeds=range(5), verbose=True):
    ts, X, y = make_synthetic()
    fp = make_fit_predict(X, y)
    settings = [("row split, random benign (leaky)", dict(mode="row", benign="random")),
                ("whole bursts, random benign", dict(mode="burst", benign="random", gap=5.0)),
                ("whole bursts, time-blocked benign", dict(mode="burst", benign="block", gap=5.0)),
                ("chronological", dict(mode="chrono"))]
    table = []
    for name, kw in settings:
        res = run_protocol(ts, y, fp, seeds=seeds, fpr_targets=(2e-3, 5e-3, 1e-2), **kw)
        rows, _ = summarize(res)
        row = {(r["model"], r["rule"]): r for r in rows}
        table.append((name, row))
    if verbose:
        print(f"{'setting':40s}{'nn  MCC':>10s}{'lda MCC':>10s}   (median over {len(list(seeds))} runs, FPR target 0.5%)")
        for name, row in table:
            print(f"{name:40s}{row[('nn', 'fpr0.005')]['mcc_median']:>10.3f}{row[('lda', 'fpr0.005')]['mcc_median']:>10.3f}")
    return table
