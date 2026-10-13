"""make_split must reproduce, index for index, the split functions of the paper's experiment runner
(cicapt_runner_v4.py). The reference functions below are copied from that runner with their globals made arguments."""
import numpy as np
import aptevalkit as ak

BLOCK_S = 300.0


def ref_partition(ids, rng, props):
    nb = int(ids.max()) + 1
    sizes = np.bincount(ids, minlength=nb)
    target, counts, assign = sizes.sum() * np.asarray(props, float), np.zeros(len(props)), np.zeros(nb, dtype=int)
    for bb in rng.permutation(nb):
        c = int(np.argmax(target - counts))
        assign[bb] = c
        counts[c] += sizes[bb]
    return assign[ids]


def ref_bursts(t, gap):
    order = np.argsort(t, kind="stable")
    new = np.concatenate([[True], np.diff(t[order]) > gap])
    ids = np.empty(len(t), dtype=int)
    ids[order] = np.cumsum(new) - 1
    return ids


def ref_chrono_attack_split(ts_att):
    ids = ref_bursts(ts_att, 5.0)
    nb = int(ids.max()) + 1
    sizes = np.bincount(ids, minlength=nb)
    start = np.array([ts_att[ids == k].min() for k in range(nb)])
    order = np.argsort(start)
    mid = (np.cumsum(sizes[order]) - sizes[order] / 2) / sizes.sum()
    part = np.where(mid < 0.6, 0, np.where(mid < 0.8, 1, 2))
    assign = np.empty(nb, dtype=int)
    assign[order] = part
    return assign[ids], (float(start[order][part == 1].min()), float(start[order][part == 2].min()))


def ref_benign_split(rng, ts, ben_all, benign, cuts=None):
    if cuts is not None:
        tb = ts[ben_all]
        return ben_all[tb < cuts[0]], ben_all[(tb >= cuts[0]) & (tb < cuts[1])], ben_all[tb >= cuts[1]]
    if benign == "random":
        p = rng.permutation(ben_all)
        c1, c2 = int(.6 * len(p)), int(.8 * len(p))
        return np.sort(p[:c1]), np.sort(p[c1:c2]), np.sort(p[c2:])
    blk = ((ts[ben_all] - ts.min()) // BLOCK_S).astype(int)
    sp = ref_partition(blk, rng, (.6, .2, .2))
    return ben_all[sp == 0], ben_all[sp == 1], ben_all[sp == 2]


def make_data(seed, shuffle_attack_order):
    rng = np.random.default_rng(seed)
    n_b = 30000
    tb = rng.uniform(1000, 4e5, n_b)
    ts_parts, stage_parts = [tb], []
    stages = np.array(["discovery", "collection", "exfiltration", "cleanup"])
    starts = np.sort(rng.uniform(1000, 4e5, 70))
    st = []
    for k, s in enumerate(starts):
        m = int(rng.integers(3, 25))
        ts_parts.append(s + rng.uniform(0.2, 3.0, m).cumsum())
        st += [stages[(k * 4) // 70]] * m
    ts = np.concatenate(ts_parts)
    y = np.concatenate([np.zeros(n_b, int), np.ones(len(ts) - n_b, int)])
    att_rows = np.arange(n_b, len(ts))
    stage = np.array(st)
    if shuffle_attack_order:
        perm = rng.permutation(len(att_rows))
        att_rows, stage = att_rows[perm], stage[perm]
    # shuffle benign flows into the row order as well (rows are not time-sorted in the real data)
    return ts, y, att_rows, stage


def _same(idx, attacks_by_part, benign_by_part):
    for k, name in enumerate(("train", "val", "test")):
        got = idx[name]
        want = np.sort(np.concatenate([attacks_by_part[k], benign_by_part[k]]))
        assert np.array_equal(np.sort(got), want), f"{name} differs"


def test_equivalence_with_runner_all_modes():
    for data_seed in (0, 1):
        for shuffle in (False, True):
            ts, y, att_rows, stage = make_data(data_seed, shuffle)
            ts_att = ts[att_rows]
            ben_all = np.where(y == 0)[0]
            for seed in range(4):
                # row / random5 / random120 with both benign rules
                for mode, gap, ids_fn in (("row", 5.0, lambda t: np.arange(len(t))),
                                          ("burst", 5.0, lambda t: ref_bursts(t, 5.0)),
                                          ("burst", 120.0, lambda t: ref_bursts(t, 120.0))):
                    for benign in ("random", "block"):
                        rng = np.random.default_rng(seed)
                        sp = ref_partition(ids_fn(ts_att), rng, (.6, .2, .2))
                        btr, bva, bte = ref_benign_split(rng, ts, ben_all, benign)
                        idx = ak.make_split(ts, y, mode, gap, benign, rng=np.random.default_rng(seed),
                                            attack_idx=att_rows)
                        _same(idx, [att_rows[sp == k] for k in range(3)], [btr, bva, bte])
                # chronological (deterministic)
                sp, cuts = ref_chrono_attack_split(ts_att)
                btr, bva, bte = ref_benign_split(None, ts, ben_all, "block", cuts)
                idx = ak.make_split(ts, y, "chrono", 5.0, attack_idx=att_rows)
                _same(idx, [att_rows[sp == k] for k in range(3)], [btr, bva, bte])
                # leave one stage out (first stage drawn after the benign split, as in the runner)
                for s in ("collection", "cleanup"):
                    for benign in ("block", "random"):
                        rng = np.random.default_rng(seed)
                        btr, bva, bte = ref_benign_split(rng, ts, ben_all, benign)
                        te = stage == s
                        sp = np.full(len(stage), 2)
                        sp[~te] = ref_partition(ref_bursts(ts_att[~te], 5.0), rng, (.75, .25))
                        idx = ak.make_split(ts, y, "loso", 5.0, benign, rng=np.random.default_rng(seed),
                                            attack_idx=att_rows, groups=stage, held_out=s)
                        _same(idx, [att_rows[sp == k] for k in range(3)], [btr, bva, bte])
