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


def feature_drift_kl_signal(X, chunk_size: int, n_bins: int = 10, eps: float = 1e-6) -> np.ndarray:
    """Chunk-over-chunk feature-distribution drift signal: for each chunk after the first, the mean
    (over features) KL divergence between the current chunk's empirical distribution and the
    immediately preceding chunk's, using fixed-width histograms binned from each feature's full-
    stream range (so chunk-to-chunk histograms are directly comparable).

    Unlike the residual-loss signal (naturally a per-instance quantity), a feature distribution is
    inherently a window/batch-level property -- there's no single-instance notion of "distribution
    drift". This signal is therefore one scalar PER CHUNK, not per row, and BOCPD consuming it runs
    at chunk granularity (see Phase 5's ablation report for how this is used).

    Returns an array of length `n_chunks = len(X) // chunk_size`, with signal[0] = 0.0 (no prior
    chunk to compare against).
    """
    import pandas as pd  # local import: signals.py otherwise has no pandas dependency

    n_chunks = len(X) // chunk_size
    if n_chunks < 2:
        raise ValueError(f"Need at least 2 full chunks of size {chunk_size} to compute drift, got {n_chunks}")

    bin_edges = {col: np.histogram_bin_edges(X[col].to_numpy(), bins=n_bins) for col in X.columns}

    def chunk_histograms(chunk: "pd.DataFrame") -> list[np.ndarray]:
        hists = []
        for col in X.columns:
            h, _ = np.histogram(chunk[col].to_numpy(), bins=bin_edges[col])
            h = h.astype(np.float64) + eps
            h /= h.sum()
            hists.append(h)
        return hists

    signal = np.zeros(n_chunks, dtype=np.float64)
    prev_hists = chunk_histograms(X.iloc[0:chunk_size])
    for i in range(1, n_chunks):
        chunk = X.iloc[i * chunk_size:(i + 1) * chunk_size]
        hists = chunk_histograms(chunk)
        kl_per_feature = [float(np.sum(p * np.log(p / q))) for p, q in zip(hists, prev_hists)]
        signal[i] = float(np.mean(kl_per_feature))
        prev_hists = hists

    return signal
