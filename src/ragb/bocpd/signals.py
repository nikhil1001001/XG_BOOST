"""Monitored signals for BOCPD: rolling log-loss residuals of the current ensemble, and sliding-window feature-drift KL signal.

Phase 2 implements and validates the residual-loss signal only (per the roadmap). The feature-drift
KL signal is a Phase 5 ablation (detection-signal choice: residual-loss vs. feature-drift KL) and is
implemented then, once there's an actual ablation consuming it -- stubbed here for now rather than
built ahead of need.
"""

from __future__ import annotations

import numpy as np


def binary_log_loss(y_true: np.ndarray, y_pred_proba: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Per-instance binary cross-entropy loss, clipped for numerical stability. This is the raw
    residual-loss signal named in Section 3 point 1 -- noisy per instance (a Bernoulli draw), so
    BOCPD is normally fed a smoothed version of it (see `rolling_mean`), not this directly.
    """
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred_proba, dtype=float), eps, 1 - eps)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling mean, same length as input (edge-shrunk window at the start) -- used to turn
    the noisy per-instance residual-loss signal into something closer to the smooth real-valued
    signal BOCPD's Normal-Inverse-Gamma predictive model is designed for.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.empty(n, dtype=float)
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    for i in range(n):
        lo = max(0, i - window + 1)
        out[i] = (cumsum[i + 1] - cumsum[lo]) / (i + 1 - lo)
    return out


def feature_drift_kl_signal(*args, **kwargs):
    """Sliding-window feature-distribution drift signal (KL divergence between a reference window and
    a current window). Not yet implemented -- lands in Phase 5 when the residual-loss-vs-feature-drift
    ablation is actually run (Section 5, Phase 5 ablations).
    """
    raise NotImplementedError("feature_drift_kl_signal is implemented in Phase 5 (signal-choice ablation)")
