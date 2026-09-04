"""Evaluation metrics: PR-AUC over time, detection lag, false-alarm rate, TS-AUC pairwise ranking.

Implemented incrementally as later phases need them (Section 0's phase-by-phase rule) -- Phase 1
only needs windowed PR-AUC; detection lag / false-alarm rate land in Phase 2, TS-AUC lands when the
Section 6 secondary-metric evaluation is actually run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Precision-Recall AUC (average precision). Returns NaN if a window has only one class present
    (undefined PR-AUC), so callers can distinguish "bad score" from "not computable here".
    """
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def windowed_pr_auc(y_true: np.ndarray, y_score: np.ndarray, window_size: int) -> pd.DataFrame:
    """PR-AUC computed over consecutive non-overlapping windows of the stream, so accuracy can be
    tracked over time (drift-adjusted, per Section 8) instead of collapsed into one scalar.

    Returns a DataFrame with columns: window_start, window_end, n, n_pos, pr_auc.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    rows = []
    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        yt = y_true[start:end]
        ys = y_score[start:end]
        rows.append({
            "window_start": start,
            "window_end": end,
            "n": end - start,
            "n_pos": int(yt.sum()),
            "pr_auc": pr_auc(yt, ys),
        })
    return pd.DataFrame(rows)
