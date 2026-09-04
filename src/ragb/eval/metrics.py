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


def detect_events_from_probability_trace(probability_trace: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Collapses a per-timestep probability trace (e.g. BOCPD's `post_break_weight`, NOT the raw
    `changepoint_probability`, which is pinned to the constant hazard rate and carries no signal --
    see `bocpd/detector.py`) into discrete detection events: the first timestep of each contiguous
    run where the trace >= threshold. Bursty exceedances near a real break (several consecutive
    high-probability steps) collapse to one event rather than being counted as multiple
    detections/false-alarms.
    """
    above = np.asarray(probability_trace) >= threshold
    if not above.any():
        return np.array([], dtype=int)
    starts = np.where(above & ~np.concatenate(([False], above[:-1])))[0]
    return starts


def detection_lag_and_false_alarms(
    true_breakpoints: np.ndarray,
    detected_events: np.ndarray,
    max_lag: int,
    n_timesteps: int,
) -> dict:
    """Matches each true breakpoint to the earliest unmatched detection within [bp, bp + max_lag].

    Returns a dict with: n_true_breaks, n_detected (true breaks that got a matching detection),
    n_missed, mean_lag (over matched breaks only), n_false_alarms (detections matched to no true
    break), false_alarm_rate (false alarms per 1000 timesteps).
    """
    true_breakpoints = np.asarray(true_breakpoints, dtype=int)
    detected_events = np.asarray(detected_events, dtype=int)
    used = np.zeros(len(detected_events), dtype=bool)

    lags = []
    n_missed = 0
    for bp in true_breakpoints:
        candidates = np.where((detected_events >= bp) & (detected_events <= bp + max_lag) & (~used))[0]
        if len(candidates) == 0:
            n_missed += 1
            continue
        best = candidates[np.argmin(detected_events[candidates] - bp)]
        used[best] = True
        lags.append(int(detected_events[best] - bp))

    n_false_alarms = int((~used).sum())
    return {
        "n_true_breaks": len(true_breakpoints),
        "n_detected": len(lags),
        "n_missed": n_missed,
        "mean_lag": float(np.mean(lags)) if lags else float("nan"),
        "n_false_alarms": n_false_alarms,
        "false_alarm_rate_per_1000": n_false_alarms / n_timesteps * 1000 if n_timesteps else float("nan"),
    }


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
