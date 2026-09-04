"""Naive sliding-window retrain baseline — stronger heuristic than periodic retrain, still changepoint-unaware."""

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


def run_sliding_window_retrain(
    X: pd.DataFrame,
    y: pd.Series,
    initial_train_frac: float = 0.2,
    retrain_every: int = 1000,
    window_size_rows: int = 4000,
    xgb_params: dict | None = None,
    seed: int = 42,
    window_size: int = 500,
) -> dict:
    """Like `periodic_retrain`, but each retrain cycle fits only on the most recent `window_size_rows`
    instead of all history seen so far -- a common "just forget old data" heuristic that's stronger
    than a stale static model but still has no notion of *where* a regime boundary actually is (it
    discards a fixed amount of history regardless of whether a break just happened or not).
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
        "sliding_window_retrain: initial_train=%d rows, retrain_every=%d, window_size_rows=%d, %d retrain cycles, params=%s",
        split, retrain_every, window_size_rows, len(chunk_starts), params,
    )

    train_end = split
    for chunk_start in tqdm(chunk_starts, desc="sliding_window_retrain: walk-forward cycles", mininterval=1.0):
        train_start = max(0, train_end - window_size_rows)
        t0 = time.time()
        model = xgb.XGBClassifier(**params)
        model.fit(X.iloc[train_start:train_end], y.iloc[train_start:train_end])
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
        "sliding_window_retrain: done, %d retrains, total_train_time=%.2fs, mean PR-AUC over eval windows=%.4f",
        n_retrains, total_train_time, pr_auc_df["pr_auc"].dropna().mean(),
    )

    return {
        "scores": scores,
        "pr_auc_over_time": pr_auc_df,
        "metadata": {
            "method": "sliding_window_retrain",
            "train_rows": split,
            "eval_rows": n - split,
            "retrain_every": retrain_every,
            "window_size_rows": window_size_rows,
            "n_retrains": n_retrains,
            "total_boosting_rounds": total_rounds,
            "train_time_sec": total_train_time,
            "params": params,
        },
    }
