import json
import warnings
import numpy as np
import pytest
import aptevalkit as ak
from aptevalkit.cli import selftest, _summarize_dir


def test_selftest():
    selftest()


def test_fpr_threshold_hits_target_and_warns_when_unresolvable():
    rng = np.random.default_rng(0)
    b = rng.normal(size=200_000)
    assert abs(int((b >= ak.fpr_threshold(b, 1e-3)).sum()) - 200) <= 3
    with pytest.warns(RuntimeWarning):
        ak.fpr_threshold(b[:100], 1e-4)


def test_saturation_detects_probability_pile_up():
    s = np.concatenate([np.full(900, 1.0), np.linspace(0, 0.99, 100)])
    assert ak.saturation_report(s)["share_within_tol_of_max"] == 0.9


def test_burst_split_keeps_bursts_whole_and_is_reproducible():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 1e5, 4000))
    y = (rng.random(4000) < 0.1).astype(int)
    a = ak.make_split(t, y, "burst", 5.0, "block", rng=np.random.default_rng(3))
    b = ak.make_split(t, y, "burst", 5.0, "block", rng=np.random.default_rng(3))
    for k in a:
        assert np.array_equal(a[k], b[k])
    assert len(np.concatenate(list(a.values()))) == len(t)


def test_chrono_split_is_ordered_and_needs_attacks():
    rng = np.random.default_rng(1)
    t = rng.uniform(0, 1e5, 4000)
    y = (rng.random(4000) < 0.05).astype(int)
    s = ak.make_split(t, y, "chrono")
    assert t[s["train"]].max() <= t[s["val"]].min() <= t[s["val"]].max() <= t[s["test"]].min()
    with pytest.raises(ValueError):
        ak.make_split(t, np.zeros(4000, int), "chrono")


def test_loso_holds_out_the_whole_group():
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 1e5, 5000))
    y = (rng.random(5000) < 0.1).astype(int)
    att = np.where(y == 1)[0]
    stage = np.array(["a", "b", "c"])[rng.integers(0, 3, len(att))]
    s = ak.make_split(t, y, "loso", 5.0, "block", rng=np.random.default_rng(0), groups=stage, held_out="b")
    in_test = set(s["test"])
    assert all((r in in_test) == (g == "b") for r, g in zip(att, stage))


def test_run_protocol_schema_and_threshold_uses_no_test_data(tmp_path):
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 1e5, 6000))
    y = (rng.random(6000) < 0.1).astype(int)
    x = rng.normal(size=6000) + 2 * y

    def fit_predict(tr, va, te, seed):
        return {"toy": (x[va], x[te])}

    res = ak.run_protocol(t, y, fit_predict, mode="row", benign="random", seeds=range(3),
                          fpr_targets=(1e-2, 2e-2), out_dir=str(tmp_path))
    assert len(res) == 3 and set(res[0]["res"]["toy"]) == {"fpr0.01", "fpr0.02", "mccrule"}
    assert (tmp_path / "seed00.json").exists()
    rows, paired = ak.summarize(res)
    assert rows and not paired
    _summarize_dir(str(tmp_path), None)


def test_paired_summary_flags_a_real_difference():
    rng = np.random.default_rng(0)
    a = 0.8 + rng.normal(0, .01, 20)
    b = 0.6 + rng.normal(0, .01, 20)
    s = ak.paired_summary(a, b, n_train=60, n_test=20, n_comparisons=4)
    assert s["ci_runs"][0] > 0.15 and s["a_better"] == 20 and s["p_bonferroni"] <= 1.0
