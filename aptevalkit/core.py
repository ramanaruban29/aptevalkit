"""Core helpers for leakage-aware, repeated-run evaluation of flow-based APT detectors."""
import warnings
import numpy as np

CHECKLIST = [
    ("repeats", "Whole pipeline repeated with redrawn splits; mean, sd and range reported"),
    ("bursts", "Attacks grouped into bursts before splitting; burst gap varied"),
    ("threshold", "Threshold rule and FPR targets fixed before looking at test data; 3+ operating points"),
    ("tuning", "Every baseline given the same tuning budget; main model settings stated"),
    ("paired", "Paired differences with intervals; number of comparisons stated"),
    ("ports", "Ablation removing port and address-like features"),
    ("stages", "Stages and bursts in test reported; recall per burst or stage"),
    ("release", "Code, split indices and saved scores released"),
]


def bursts_from_gap(ts, gap):
    """Group events whose consecutive gaps are <= gap seconds; returns a burst id per event."""
    ts = np.asarray(ts, float)
    order = np.argsort(ts, kind="stable")
    new = np.concatenate([[True], np.diff(ts[order]) > gap])
    ids = np.empty(len(ts), dtype=int)
    ids[order] = np.cumsum(new) - 1
    return ids


def time_block_ids(ts, block_s=300.0):
    """Fixed-length time blocks, for splitting benign flows without scattering near-duplicates."""
    b = ((np.asarray(ts, float) - np.min(ts)) // block_s).astype(np.int64)
    return np.unique(b, return_inverse=True)[1]


def partition_groups(group_ids, rng, props=(0.6, 0.2, 0.2)):
    """Assign whole groups (bursts or blocks) to partitions, greedily by size; returns a partition index per row."""
    g = np.asarray(group_ids)
    ng = int(g.max()) + 1
    sizes = np.bincount(g, minlength=ng)
    target = sizes.sum() * np.asarray(props, float)
    counts = np.zeros(len(props))
    assign = np.zeros(ng, dtype=int)
    for k in rng.permutation(ng):
        c = int(np.argmax(target - counts))
        assign[k] = c
        counts[c] += sizes[k]
    return assign[g]


def mcc(tp, fp, fn, tn):
    d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / d) if d > 0 else 0.0


def fpr_threshold(val_benign, fpr):
    """Threshold giving the target false-positive rate on validation benign flows; uses no attack labels.

    Warns when the validation set is too small to resolve the target (fewer than one expected false positive).
    """
    val_benign = np.asarray(val_benign, float)
    if len(val_benign) * fpr < 1.0:
        warnings.warn(f"validation benign set ({len(val_benign)}) too small to resolve FPR {fpr:g}; "
                      "the threshold is the maximum validation score", RuntimeWarning)
    return float(np.quantile(val_benign, 1.0 - fpr))


def metrics_at(thr, benign, attack):
    benign = np.asarray(benign)
    attack = np.asarray(attack)
    tp = int((attack >= thr).sum())
    fp = int((benign >= thr).sum())
    fn = len(attack) - tp
    tn = len(benign) - fp
    return dict(mcc=mcc(tp, fp, fn, tn), recall=tp / max(len(attack), 1), fp=fp, tp=tp, thr=float(thr))


def tie_report(benign, attack):
    """Share of flows tied at the maximum score. A large share means FPR targets below it cannot be met."""
    benign = np.asarray(benign)
    attack = np.asarray(attack)
    allv = np.concatenate([benign, attack])
    mx = allv.max()
    return dict(n=int(len(allv)), unique=int(len(np.unique(allv))),
                benign_at_max=float((benign == mx).mean()), attack_at_max=float((attack == mx).mean()))


def saturation_report(scores, tol=1e-6):
    """Detect probability saturation: share of scores within tol of the maximum, and of the top-1 / 99.9 percent.

    Classifiers trained with benign undersampling can push most benign scores to ~1.0. If the threshold for a strict
    FPR target then lands within float noise of 1.0, the target cannot be met and recall collapses.
    """
    s = np.asarray(scores, float)
    mx = s.max()
    return dict(n=int(len(s)), max=float(mx), share_within_tol_of_max=float((s >= mx - tol).mean()),
                unique=int(len(np.unique(s))), q999=float(np.quantile(s, 0.999)))


def paired_summary(a, b, n_train, n_test, n_boot=10000, seed=0, n_comparisons=1):
    """Paired difference of per-run scores (a - b): run bootstrap interval plus the corrected resampled t-test
    (Nadeau and Bengio 2003). p_bonferroni multiplies the corrected p by the number of comparisons."""
    d = np.asarray(a, float) - np.asarray(b, float)
    J = len(d)
    bs = np.random.default_rng(seed).choice(d, (n_boot, J)).mean(1)
    out = dict(runs=J, mean=float(d.mean()), sd=float(d.std(ddof=1)) if J > 1 else float("nan"),
               ci_runs=(float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))),
               a_better=int((d > 0).sum()), n_comparisons=int(n_comparisons))
    if J > 1 and d.std(ddof=1) > 0:
        t = d.mean() / np.sqrt((1.0 / J + n_test / n_train) * d.var(ddof=1))
        try:
            from scipy import stats
            p = float(2 * stats.t.sf(abs(t), J - 1))
        except Exception:
            p = float("nan")
        out.update(t_corrected=float(t), p_corrected=p, p_bonferroni=min(1.0, p * n_comparisons))
    return out


def burst_bootstrap_recall_diff(det_a, det_b, burst_ids, n_boot=5000, seed=0):
    """Resample test bursts; interval for the recall difference of two detectors at fixed thresholds."""
    da, db, ids = np.asarray(det_a, float), np.asarray(det_b, float), np.asarray(burst_ids)
    ub = np.unique(ids)
    sa = np.array([da[ids == u].sum() for u in ub])
    sb = np.array([db[ids == u].sum() for u in ub])
    n = np.array([(ids == u).sum() for u in ub])
    idx = np.random.default_rng(seed).integers(0, len(ub), (n_boot, len(ub)))
    diff = (sa[idx].sum(1) - sb[idx].sum(1)) / n[idx].sum(1)
    return float(diff.mean()), (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)))


def print_checklist(done):
    for key, text in CHECKLIST:
        print(f"[{'x' if done.get(key) else ' '}] {text}")
