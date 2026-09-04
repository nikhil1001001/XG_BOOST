"""Periodic full retrain baseline (e.g. weekly) — common industry-default retrain schedule."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

from ragb.baselines.static_xgb import DEFAULT_XGB_PARAMS
from ragb.eval.metrics import windowed_pr_auc
from ragb.utils.logging_config import get_logger

logger = get_logger(__name__)


def run_periodic_retrain(
    X: pd.DataFrame,
    y: pd.Series,
    initial_train_frac: float = 0.2,
    retrain_every: int = 1000,
    xgb_params: dict | None = None,
    seed: int = 42,
    window_size: int = 500,
) -> dict:
    """Walk-forward periodic full retrain: starting from an initial training window, predict the
    next `retrain_every` rows with the current model, then refit from scratch on ALL data seen so
    far (cumulative, not sliding -- the sliding variant is `sliding_window_retrain.py`) before
    scoring the next chunk. Tracks amortized compute cost (Section 8) via n_retrains and total
    boosting rounds fit, for later comparison against RAGB's spawn-triggered training.
    """
    params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {}), "random_state": seed}
    n = len(X)
    split = int(n * initial_train_frac)

    scores = np.full(n, np.nan)
    n_retrains = 0
    total_rounds = 0
    total_train_time = 0.0

    chunk_starts = list(range(split, n, retrain_every))
    logger.info(
        "periodic_retrain: initial_train=%d rows, retrain_every=%d, %d retrain cycles, params=%s",
        split, retrain_every, len(chunk_starts), params,
    )

    train_end = split
    for chunk_start in tqdm(chunk_starts, desc="periodic_retrain: walk-forward cycles", mininterval=1.0):
        t0 = time.time()
        model = xgb.XGBClassifier(**params)
        model.fit(X.iloc[:train_end], y.iloc[:train_end])
        total_train_time += time.time() - t0
        n_retrains += 1
        total_rounds += params["n_estimators"]

        chunk_end = min(chunk_start + retrain_every, n)
        scores[chunk_start:chunk_end] = model.predict_proba(X.iloc[chunk_start:chunk_end])[:, 1]
        train_end = chunk_end

    pr_auc_df = windowed_pr_auc(y.iloc[split:].to_numpy(), scores[split:], window_size)
    pr_auc_df["window_start"] += split
    pr_auc_df["window_end"] += split

    logger.info(
        "periodic_retrain: done, %d retrains, total_train_time=%.2fs, mean PR-AUC over eval windows=%.4f",
        n_retrains, total_train_time, pr_auc_df["pr_auc"].dropna().mean(),
    )

    return {
        "scores": scores,
        "pr_auc_over_time": pr_auc_df,
        "metadata": {
            "method": "periodic_retrain",
            "train_rows": split,
            "eval_rows": n - split,
            "retrain_every": retrain_every,
            "n_retrains": n_retrains,
            "total_boosting_rounds": total_rounds,
            "train_time_sec": total_train_time,
            "params": params,
        },
    }
