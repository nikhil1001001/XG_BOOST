"""Single warm-started XGBoost sub-ensemble using BOCPD soft weights end-to-end (Phase 3 milestone).

Design: one continuously warm-started booster (`xgb_model=` chaining, never restarted from scratch)
processes the stream chunk by chunk. For each new chunk: (1) score it with the current booster to get
a residual-loss signal, (2) feed that signal into a single continuously-running BOCPD detector
(one online detector for the whole stream, not reset per chunk), (3) derive a per-instance soft
weight for every instance in the chunk from BOCPD's current run-length posterior, and (4) add a
handful of new boosting rounds fit on the chunk with those weights via the custom soft-weighted
objective (Section 3 point 4), warm-started from the existing booster.

The soft weight (Section 3's "P(instance i is post-break)") is operationalized as a survival
probability: for an instance observed `d` steps before "now" (the end of the current chunk), its
weight is P(current run length >= d | x_1:now) -- i.e. the posterior probability the active regime
extends back far enough to cover that instance. This is 1 for instances at "now", decays smoothly for
older instances, and decays sharply right around a suspected changepoint -- exactly the "soft-assign
instances near a suspected changepoint instead of a brittle hard split" behavior Section 3 asks for,
and it falls directly out of the run-length posterior BOCPD already computes (no extra machinery).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

from ragb.bocpd.detector import BOCPD
from ragb.bocpd.signals import binary_log_loss, rolling_mean
from ragb.boosting.custom_objective import make_soft_weighted_logloss_obj
from ragb.eval.metrics import windowed_pr_auc
from ragb.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_XGB_PARAMS = dict(
    max_depth=4,
    eta=0.1,
    tree_method="hist",
    objective="binary:logistic",
    base_score=0.5,
)


def survival_weights(run_length_posterior: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """w_i = P(r_now >= d_i | x_1:now) for each instance at distance d_i (timesteps before "now").

    distances >= len(run_length_posterior) get weight 0: BOCPD's pruned posterior carries ~0 mass
    that far back, so those instances are treated as belonging to a definitely-ended regime.
    """
    survival = np.cumsum(run_length_posterior[::-1])[::-1]  # survival[k] = sum_{j>=k} posterior[j]
    L = len(survival)
    weights = np.zeros(len(distances), dtype=np.float64)
    in_range = distances < L
    weights[in_range] = survival[distances[in_range]]
    return weights


def run_single_expert(
    X: pd.DataFrame,
    y: pd.Series,
    initial_train_frac: float = 0.2,
    chunk_size: int = 500,
    hazard_rate: float = 0.002,
    smoothing_window: int = 20,
    initial_boost_rounds: int = 100,
    boost_rounds_per_chunk: int = 20,
    xgb_params: dict | None = None,
    seed: int = 42,
    window_size: int = 500,
) -> dict:
    params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {}), "seed": seed}
    n = len(X)
    split = int(n * initial_train_frac)

    logger.info(
        "single_expert: initial training on first %d/%d rows, %d rounds, params=%s",
        split, n, initial_boost_rounds, params,
    )
    t0 = time.time()
    dtrain_init = xgb.DMatrix(X.iloc[:split], label=y.iloc[:split])
    booster = xgb.train(params, dtrain_init, num_boost_round=initial_boost_rounds)
    total_train_time = time.time() - t0
    total_rounds = initial_boost_rounds

    detector = BOCPD(hazard_rate=hazard_rate)
    scores = np.full(n, np.nan)
    scores[:split] = booster.predict(dtrain_init)

    chunk_starts = list(range(split, n, chunk_size))
    weight_stats = []

    logger.info(
        "single_expert: %d warm-started chunks, chunk_size=%d, hazard_rate=%.5f, boost_rounds_per_chunk=%d",
        len(chunk_starts), chunk_size, hazard_rate, boost_rounds_per_chunk,
    )

    for chunk_start in tqdm(chunk_starts, desc="single_expert: warm-started chunks", mininterval=1.0):
        chunk_end = min(chunk_start + chunk_size, n)
        X_chunk = X.iloc[chunk_start:chunk_end]
        y_chunk = y.iloc[chunk_start:chunk_end]

        chunk_scores = booster.predict(xgb.DMatrix(X_chunk))
        scores[chunk_start:chunk_end] = chunk_scores

        residual = binary_log_loss(y_chunk.to_numpy(), chunk_scores)
        smoothed = rolling_mean(residual, smoothing_window)
        for x in smoothed:
            detector.update(float(x))

        posterior = detector.run_length_posterior
        instance_times = np.arange(chunk_start, chunk_end)
        distances = (chunk_end - 1) - instance_times
        weights = survival_weights(posterior, distances)
        weight_stats.append((chunk_start, float(weights.min()), float(weights.mean()), float(weights.max())))
        logger.debug(
            "chunk[%d:%d]: weight min=%.4f mean=%.4f max=%.4f",
            chunk_start, chunk_end, weights.min(), weights.mean(), weights.max(),
        )

        t0 = time.time()
        dchunk = xgb.DMatrix(X_chunk, label=y_chunk)
        obj = make_soft_weighted_logloss_obj(weights)
        booster = xgb.train(params, dchunk, num_boost_round=boost_rounds_per_chunk, xgb_model=booster, obj=obj)
        total_train_time += time.time() - t0
        total_rounds += boost_rounds_per_chunk

    pr_auc_df = windowed_pr_auc(y.iloc[split:].to_numpy(), scores[split:], window_size)
    pr_auc_df["window_start"] += split
    pr_auc_df["window_end"] += split

    logger.info(
        "single_expert: done, %d chunks, total_rounds=%d, total_train_time=%.2fs, mean PR-AUC over eval windows=%.4f",
        len(chunk_starts), total_rounds, total_train_time, pr_auc_df["pr_auc"].dropna().mean(),
    )

    return {
        "scores": scores,
        "pr_auc_over_time": pr_auc_df,
        "weight_stats": pd.DataFrame(weight_stats, columns=["chunk_start", "weight_min", "weight_mean", "weight_max"]),
        "metadata": {
            "method": "single_expert",
            "train_rows": split,
            "eval_rows": n - split,
            "n_chunks": len(chunk_starts),
            "chunk_size": chunk_size,
            "hazard_rate": hazard_rate,
            "smoothing_window": smoothing_window,
            "boost_rounds_per_chunk": boost_rounds_per_chunk,
            "n_retrains": len(chunk_starts) + 1,  # initial fit + one warm-start update per chunk
            "total_boosting_rounds": total_rounds,
            "train_time_sec": total_train_time,
            "params": params,
        },
    }
