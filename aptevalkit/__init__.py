"""aptevalkit: leakage-aware, repeated-run evaluation of flow-based APT detectors."""
from .core import (CHECKLIST, bursts_from_gap, time_block_ids, partition_groups, mcc, fpr_threshold, metrics_at,
                   tie_report, saturation_report, paired_summary, burst_bootstrap_recall_diff, print_checklist)
from .protocol import make_split, evaluate_scores, run_protocol, summarize

__version__ = "0.1.1"
__all__ = ["CHECKLIST", "bursts_from_gap", "time_block_ids", "partition_groups", "mcc", "fpr_threshold", "metrics_at",
           "tie_report", "saturation_report", "paired_summary", "burst_bootstrap_recall_diff", "print_checklist",
           "make_split", "evaluate_scores", "run_protocol", "summarize"]
