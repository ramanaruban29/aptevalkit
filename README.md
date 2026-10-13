# aptevalkit

Leakage-aware, repeated-run evaluation for flow-based APT detectors. It accompanies the paper
*"What a Single Run Hides: Repeated-Run Evaluation of an Autoencoder + LightGBM APT Detector on Two Network Flow
Datasets"* (CICAPT-IIoT and DAPT2020).

A single train/test split of an APT dataset gives one number whose spread across redrawn splits (about 0.2 MCC on
CICAPT-IIoT) is larger than most differences between published detectors. The toolkit makes the protocol easy to
apply to **any** detector: you supply the model, it supplies the splits, thresholds, repeats and paired statistics.

## Install

```bash
pip install .            # numpy only
pip install ".[stats]"   # adds scipy for the corrected t-test p-values
python -m aptevalkit selftest
python -m aptevalkit demo
```

## The protocol (the "leakage ladder")

Each rung is a way of splitting flows into train / validation / test. Rungs lower on the list leak less:

| rung | `make_split(...)` | what it controls |
|---|---|---|
| row split | `mode="row"` | every attack flow is its own group (leaky reference: burst-mates sit on both sides) |
| whole bursts | `mode="burst", gap=5` or `120` | attack flows separated by <= `gap` s are one burst; bursts are never split |
| time-blocked benign | `benign="block"` | benign flows assigned in 5-minute blocks, not one by one (`benign="random"` is the optimistic alternative) |
| chronological | `mode="chrono"` | attack bursts ordered by start time and cut 60/20/20; benign flows cut at the same times (on datasets where stage and time are confounded this is "unseen stages") |
| leave one group out | `mode="loso", groups=..., held_out=...` | all attacks of one stage or activity are the test set; the rest is split by bursts |

Also fixed by the protocol: thresholds come from **validation benign** scores at stated FPR targets (no attack labels, no
test data) plus an MCC-optimal rule; the whole pipeline (scaler, autoencoder, classifier) is refitted on every
repeat; paired differences are reported with a run-bootstrap interval, the Nadeau-Bengio corrected resampled t-test,
a Bonferroni multiplier for the number of comparisons, and sign counts.

## Use it on your detector

```python
import numpy as np, aptevalkit as ak

# ts: timestamps (s), y: 0 benign / 1 attack, X: your features
def fit_predict(train_idx, val_idx, test_idx, seed):
    # refit EVERYTHING that sees training data here (scaler, autoencoder, classifier)
    model = MyDetector(seed=seed).fit(X[train_idx], y[train_idx])
    return {"mydetector": (model.score(X[val_idx]), model.score(X[test_idx]))}   # higher = more anomalous

results = ak.run_protocol(ts, y, fit_predict, mode="burst", gap=5.0, benign="block",
                          seeds=range(20), fpr_targets=(2e-5, 4e-5, 8e-5), out_dir="runs/burst5_block")
rows, paired = ak.summarize(results, deployed="mydetector")
```

Repeat with `mode="row"`, `gap=120`, `benign="random"`, `mode="chrono"` to build the ladder. Result files are one JSON per
repeat (`{"info": ..., "res": {model: {rule: {mcc, recall, fp, tp, thr}}}}`); summarize any folder from the shell:

```bash
python -m aptevalkit summarize runs/burst5_block --deployed mydetector
python -m aptevalkit summarize runs/burst5_block --deployed mydetector --exclude ae_only,iforest,ocsvm
```

`summarize` prints both the unadjusted corrected-t p (`p_raw`) and the Bonferroni-adjusted p (`p_adj` = `p_raw` x number
of comparisons). The number of comparisons is the number of models compared with the deployed one, so state your
comparison family in the paper (`--exclude` removes models, such as benign-only detectors that fail by a wide margin,
from the family).

Choose FPR targets your validation benign set can resolve: `fpr_threshold` warns when
`len(val_benign) * fpr < 1`.

## Diagnostics worth running

* `tie_report(benign_scores, attack_scores)` and `saturation_report(scores)` -- clipping scores (or classifier
  probabilities that saturate near 1.0 under benign undersampling) piles flows onto one value, so strict FPR targets
  cannot be met (the 0.01% target below returned 1000 FP instead of 10 in the self-test).
* `burst_bootstrap_recall_diff(det_a, det_b, burst_ids)` -- interval for a recall difference that resamples test
  bursts, not flows.
* `python -m aptevalkit checklist` -- the reporting checklist from the paper.

## What is and is not verified here

* The self-test and unit tests check the mechanics (bursts stay whole, partitions do not overlap, the chronological cut
  is ordered, the FPR threshold hits its target, the paired statistics run).
* `tests/test_runner_equivalence.py` contains the split functions of the paper's experiment runner
  (`cicapt_runner_v4.py`) and asserts that `make_split` returns **exactly the same train / validation / test rows**
  for row, burst (5 s and 120 s), chronological and leave-one-stage-out splits, with random and time-blocked benign
  flows, on synthetic data.
* `scripts/verify_against_saved_runs.py` re-derives the attack counts, benign sizes and test-stage mix of every seed
  from the raw timestamps and compares them with the saved result files of the paper's runs. Run it on your copy of
  the data. Confirmed: all 53 saved CICAPT-IIoT seeds (row, 5 s and 120 s bursts, random and time-blocked benign,
  chronological, leave-one-stage-out) are reproduced exactly. Not covered: the DAPT2020 runs (separate scripts) and
  leave-one-stage-out on real data beyond the saved CICAPT seeds.
* `python -m aptevalkit demo` is a **synthetic** illustration of the row-split leakage rung: a memory-based detector
  falls when whole bursts are held out while a linear detector does not. Synthetic bursts are exchangeable in time, so
  the demo does not show a chronological drop; that needs real data with stage or activity shift.
* The model scores, thresholds and metrics in the paper come from the original experiment scripts; `run_protocol`
  re-implements the protocol around your own `fit_predict`, so numbers from your detector are not expected to equal
  the paper's unless you reproduce its models too.

## Limitations

* Burst grouping assumes timestamps are reliable; datasets that reuse timestamps across hosts may need a per-host key.
* Chronological splits are only meaningful when time and attack stage are not perfectly confounded.
* Thresholds are rule-based and validation-set dependent; the protocol reports the resulting FP counts so overshoot is
  visible.

## License

MIT.
