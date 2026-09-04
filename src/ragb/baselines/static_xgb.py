"""Static XGBoost trained once and never updated ("do nothing" floor baseline)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xgboost as xgb

from ragb.eval.metrics import windowed_pr_auc
from ragb.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_XGB_PARAMS = dict(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    tree_method="hist",
    eval_metric="logloss",
)


def run_static_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    initial_train_frac: float = 0.2,
    xgb_params: dict | None = None,
    seed: int = 42,
    window_size: int = 500,
) -> dict:
    """Trains once on the first `initial_train_frac` of the stream, then scores everything after
    that with no further updates. This is the "stale model" failure mode from Section 1 -- the
    floor every other approach should beat.
    """
    params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {}), "random_state": seed}
    n = len(X)
    split = int(n * initial_train_frac)

    logger.info("static_xgb: training once on first %d/%d rows, params=%s", split, n, params)
    t0 = time.time()
    model = xgb.XGBClassifier(**params)
    model.fit(X.iloc[:split], y.iloc[:split])
    train_time = time.time() - t0

    scores = np.full(n, np.nan)
    scores[split:] = model.predict_proba(X.iloc[split:])[:, 1]

    pr_auc_df = windowed_pr_auc(y.iloc[split:].to_numpy(), scores[split:], window_size)
    pr_auc_df["window_start"] += split
    pr_auc_df["window_end"] += split

    logger.info(
        "static_xgb: done, train_time=%.2fs, mean PR-AUC over eval windows=%.4f",
        train_time, pr_auc_df["pr_auc"].dropna().mean(),
    )

    return {
        "scores": scores,
        "pr_auc_over_time": pr_auc_df,
        "metadata": {
            "method": "static_xgb",
            "train_rows": split,
            "eval_rows": n - split,
            "train_time_sec": train_time,
            "n_retrains": 1,
            "total_boosting_rounds": params["n_estimators"],
            "params": params,
        },
    }
