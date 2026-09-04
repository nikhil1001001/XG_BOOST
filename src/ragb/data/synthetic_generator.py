"""Controllable regime-switching synthetic data generator with seeded, recoverable ground-truth breakpoints (Phase 1).

Simulates a fraud-detection stream that undergoes discrete "eras": each era has its own generative
rule (a distinct sparse logistic weight vector over the same feature distribution), so a break is a
concept-drift event -- the meaning of the features changes, not their marginal distribution -- which
matches the "fraud tactics shift" framing in Section 1 of the brief and is what the residual-loss
BOCPD signal (Phase 2) is designed to pick up on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ragb.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SyntheticStreamConfig:
    n_samples: int = 20_000
    n_features: int = 10
    n_eras: int = 3
    hazard_rate: float = 1.0 / 1500.0  # per-step switch probability once min_run_length has elapsed
    min_run_length: int = 800  # avoid degenerate near-zero-length runs
    base_fraud_rate: float = 0.05  # approximate label incidence, achieved via per-era bias search
    era_sparsity: float = 0.5  # fraction of features zeroed out of each era's weight vector
    seed: int = 42


@dataclass
class SyntheticStream:
    X: pd.DataFrame
    y: pd.Series
    regime_id: np.ndarray  # active era index at each timestep, shape (n_samples,)
    breakpoints: np.ndarray  # sorted timestep indices where a NEW era's first sample appears
    era_weights: list  # per-era (weight_vector, bias) actually used, for debugging/inspection
    config: SyntheticStreamConfig = field(repr=False)


def _fit_bias_for_target_rate(X: np.ndarray, w: np.ndarray, target_rate: float) -> float:
    """Binary-search a bias term so sigmoid(X @ w + b) averages to roughly target_rate.

    Keeps every era's label incidence comparable even though eras have unrelated weight vectors --
    without this, a random w could produce a near-all-zero or near-all-one era by chance, degenerate
    for PR-AUC evaluation.
    """
    logits = X @ w
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = 1.0 / (1.0 + np.exp(-(logits + mid)))
        rate = rate.mean()
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _draw_era_weights(rng: np.random.Generator, n_features: int, sparsity: float) -> np.ndarray:
    w = rng.normal(loc=0.0, scale=2.0, size=n_features)
    mask = rng.random(n_features) < sparsity
    w[mask] = 0.0
    if np.allclose(w, 0.0):  # guard against fully-zeroed vector on unlucky sparsity draw
        w[rng.integers(0, n_features)] = rng.normal(scale=2.0)
    return w


def _draw_run_lengths(rng: np.random.Generator, n_samples: int, hazard_rate: float, min_run_length: int) -> list[int]:
    """Geometric-ish run lengths: after min_run_length steps, each further step has probability
    hazard_rate of ending the run. This mirrors the constant-hazard assumption BOCPD itself uses
    (Phase 2), so the ground truth here is a fair test of that assumption, not a mismatched one.
    """
    lengths = []
    remaining = n_samples
    while remaining > 0:
        extra = rng.geometric(hazard_rate) if hazard_rate > 0 else remaining
        run_len = min(min_run_length + extra - 1, remaining)
        run_len = max(run_len, 1)
        lengths.append(run_len)
        remaining -= run_len
    return lengths


class SyntheticRegimeGenerator:
    """Generates a regime-switching synthetic fraud stream per SyntheticStreamConfig."""

    def __init__(self, config: SyntheticStreamConfig | None = None):
        self.config = config or SyntheticStreamConfig()

    def generate(self) -> SyntheticStream:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        logger.info(
            "Generating synthetic stream: n_samples=%d n_features=%d n_eras=%d hazard_rate=%.6f seed=%d",
            cfg.n_samples, cfg.n_features, cfg.n_eras, cfg.hazard_rate, cfg.seed,
        )

        X_all = rng.normal(loc=0.0, scale=1.0, size=(cfg.n_samples, cfg.n_features))

        # Draw one distinct generative rule per era up front (seeded, so reproducible).
        era_weights: list[tuple[np.ndarray, float]] = []
        for era in range(cfg.n_eras):
            w = _draw_era_weights(rng, cfg.n_features, cfg.era_sparsity)
            b = _fit_bias_for_target_rate(X_all, w, cfg.base_fraud_rate)
            era_weights.append((w, b))

        run_lengths = _draw_run_lengths(rng, cfg.n_samples, cfg.hazard_rate, cfg.min_run_length)

        # Assign an era to each run, never repeating the immediately preceding era (a "switch" that
        # picks the same era again wouldn't be a detectable regime change).
        era_sequence = []
        prev_era = None
        for _ in run_lengths:
            choices = [e for e in range(cfg.n_eras) if e != prev_era] or list(range(cfg.n_eras))
            era = rng.choice(choices)
            era_sequence.append(era)
            prev_era = era

        regime_id = np.empty(cfg.n_samples, dtype=int)
        y = np.empty(cfg.n_samples, dtype=int)
        breakpoints = []
        cursor = 0
        for era, run_len in zip(era_sequence, run_lengths):
            if cursor > 0:
                breakpoints.append(cursor)
            w, b = era_weights[era]
            chunk = X_all[cursor:cursor + run_len]
            p = 1.0 / (1.0 + np.exp(-(chunk @ w + b)))
            y[cursor:cursor + run_len] = rng.binomial(1, p)
            regime_id[cursor:cursor + run_len] = era
            cursor += run_len

        breakpoints = np.array(breakpoints, dtype=int)
        columns = [f"f{i}" for i in range(cfg.n_features)]
        X_df = pd.DataFrame(X_all, columns=columns).astype("float32")
        y_series = pd.Series(y, name="label")

        logger.info(
            "Synthetic stream generated: %d breakpoints, overall fraud rate=%.4f",
            len(breakpoints), y_series.mean(),
        )

        return SyntheticStream(
            X=X_df,
            y=y_series,
            regime_id=regime_id,
            breakpoints=breakpoints,
            era_weights=era_weights,
            config=cfg,
        )


def generate_gradual_drift_stream(
    n_samples: int = 8000,
    n_features: int = 10,
    transition_start: int | None = None,
    transition_length: int = 3000,
    seed: int = 42,
    base_fraud_rate: float = 0.05,
) -> SyntheticStream:
    """Two generative rules (era A -> era B), but with NO discrete changepoint: the label-generating
    weight vector is linearly interpolated from w_A to w_B over `transition_length` steps starting at
    `transition_start` (default: stream midpoint minus half the transition), rather than switching
    instantaneously. Used for Phase 6's gradual-drift stress test -- Section 9 names this a known weak
    spot of changepoint-based methods generally (BOCPD has no sharp break to detect), in contrast to
    the discrete-break generator above. `breakpoints` is returned empty (there genuinely isn't one);
    `regime_id` is returned as -1 throughout the transition window (neither era A nor B applies
    cleanly) so callers can identify it if needed.
    """
    rng = np.random.default_rng(seed)
    if transition_start is None:
        transition_start = n_samples // 2 - transition_length // 2

    X_all = rng.normal(0.0, 1.0, size=(n_samples, n_features))
    w_a = _draw_era_weights(rng, n_features, sparsity=0.5)
    w_b = _draw_era_weights(rng, n_features, sparsity=0.5)
    b_a = _fit_bias_for_target_rate(X_all, w_a, base_fraud_rate)
    b_b = _fit_bias_for_target_rate(X_all, w_b, base_fraud_rate)

    y = np.empty(n_samples, dtype=int)
    regime_id = np.empty(n_samples, dtype=int)
    transition_end = transition_start + transition_length

    for t in range(n_samples):
        if t < transition_start:
            w, b, era = w_a, b_a, 0
        elif t >= transition_end:
            w, b, era = w_b, b_b, 1
        else:
            frac = (t - transition_start) / transition_length
            w = (1 - frac) * w_a + frac * w_b
            b = (1 - frac) * b_a + frac * b_b
            era = -1
        p = 1.0 / (1.0 + np.exp(-(X_all[t] @ w + b)))
        y[t] = rng.binomial(1, p)
        regime_id[t] = era

    columns = [f"f{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X_all, columns=columns).astype("float32")
    cfg = SyntheticStreamConfig(n_samples=n_samples, n_features=n_features, n_eras=2, seed=seed, base_fraud_rate=base_fraud_rate)
    return SyntheticStream(
        X=X_df,
        y=pd.Series(y, name="label"),
        regime_id=regime_id,
        breakpoints=np.array([], dtype=int),  # no discrete break, by design
        era_weights=[(w_a, b_a), (w_b, b_b)],
        config=cfg,
    )


def generate_single_changepoint(
    n_samples: int = 4000,
    breakpoint_idx: int = 2000,
    n_features: int = 10,
    seed: int = 42,
    base_fraud_rate: float = 0.05,
) -> SyntheticStream:
    """Convenience generator for the simplest possible case: exactly two eras, one known changepoint.

    Used by Phase 2's BOCPD validation (a single known changepoint at a specific location is the
    standard sanity check for a run-length posterior implementation) rather than the full multi-era
    generator above.
    """
    rng = np.random.default_rng(seed)
    X_all = rng.normal(0.0, 1.0, size=(n_samples, n_features))
    w0 = _draw_era_weights(rng, n_features, sparsity=0.5)
    w1 = _draw_era_weights(rng, n_features, sparsity=0.5)
    b0 = _fit_bias_for_target_rate(X_all, w0, base_fraud_rate)
    b1 = _fit_bias_for_target_rate(X_all, w1, base_fraud_rate)

    y = np.empty(n_samples, dtype=int)
    regime_id = np.empty(n_samples, dtype=int)
    for era, (w, b, lo, hi) in enumerate([
        (w0, b0, 0, breakpoint_idx),
        (w1, b1, breakpoint_idx, n_samples),
    ]):
        chunk = X_all[lo:hi]
        p = 1.0 / (1.0 + np.exp(-(chunk @ w + b)))
        y[lo:hi] = rng.binomial(1, p)
        regime_id[lo:hi] = era

    columns = [f"f{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X_all, columns=columns).astype("float32")
    cfg = SyntheticStreamConfig(
        n_samples=n_samples, n_features=n_features, n_eras=2, seed=seed, base_fraud_rate=base_fraud_rate,
    )
    return SyntheticStream(
        X=X_df,
        y=pd.Series(y, name="label"),
        regime_id=regime_id,
        breakpoints=np.array([breakpoint_idx]),
        era_weights=[(w0, b0), (w1, b1)],
        config=cfg,
    )
