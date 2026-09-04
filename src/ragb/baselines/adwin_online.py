"""ADWIN-triggered online warm-start boosting baseline (Idea #4) — tests whether RAGB's soft mixture beats a hard drift trigger.

Contrasts directly with RAGB: ADWIN (Bifet & Gavalda 2007, via `river`) monitors the same residual-loss
signal BOCPD would, but makes a hard binary drift/no-drift call instead of a continuous posterior.
On a detected drift, the booster is warm-started (not retrained from scratch) on the data accumulated
since the last drift point -- so this baseline isolates the effect of RAGB's core idea (soft,
continuous instance weighting vs. a discrete hard trigger) while keeping "warm-start boosting" itself
the same, since RAGB also warm-starts.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xgboost as xgb
from river.drift import ADWIN

from ragb.baselines.static_xgb import DEFAULT_XGB_PARAMS
from ragb.bocpd.signals import binary_log_loss
from ragb.eval.metrics import windowed_pr_auc
from ragb.utils.logging_config import get_logger

logger = get_logger(__name__)


def run_adwin_online(
    X: pd.DataFrame,
    y: pd.Series,
    initial_train_frac: float = 0.2,
    delta: float = 0.002,
    initial_boost_rounds: int = 100,
    boost_rounds_per_retrain: int = 50,
    xgb_params: dict | None = None,
    seed: int = 42,
    window_size: int = 500,
) -> dict:
    params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {}), "random_state": seed}
    n = len(X)
    split = int(n * initial_train_frac)

    logger.info("adwin_online: initial training on first %d/%d rows, delta=%.4f, params=%s", split, n, delta, params)
    t0 = time.time()
    booster = xgb.train(
        params_for_warm_start(params),
        xgb.DMatrix(X.iloc[:split], label=y.iloc[:split]),
        num_boost_round=initial_boost_rounds,
    )
    total_train_time = time.time() - t0
    total_rounds = initial_boost_rounds
    n_retrains = 1

    scores = np.full(n, np.nan)
    scores[:split] = booster.predict(xgb.DMatrix(X.iloc[:split]))

    drift_points = []
    pos = split
    retrain_start = split
    adwin = ADWIN(delta=delta)

    while pos < n:
        remaining_scores = booster.predict(xgb.DMatrix(X.iloc[pos:]))
        drift_offset = None
        for i, s in enumerate(remaining_scores):
            scores[pos + i] = s
            loss = float(binary_log_loss(np.array([y.iloc[pos + i]]), np.array([s]))[0])
            adwin.update(loss)
            if adwin.drift_detected:
                drift_offset = i
                break

        if drift_offset is None:
            break  # reached the end of the stream with no further drift detected

        drift_idx = pos + drift_offset
        drift_points.append(drift_idx)
        logger.debug("adwin_online: drift detected at t=%d (window since last retrain: [%d, %d])", drift_idx, retrain_start, drift_idx)

        t0 = time.time()
        dchunk = xgb.DMatrix(X.iloc[retrain_start:drift_idx + 1], label=y.iloc[retrain_start:drift_idx + 1])
        booster = xgb.train(params_for_warm_start(params), dchunk, num_boost_round=boost_rounds_per_retrain, xgb_model=booster)
        total_train_time += time.time() - t0
        total_rounds += boost_rounds_per_retrain
        n_retrains += 1

        retrain_start = drift_idx + 1
        pos = drift_idx + 1
        adwin = ADWIN(delta=delta)  # reset after acting on a detected drift, standard ADWIN usage

    pr_auc_df = windowed_pr_auc(y.iloc[split:].to_numpy(), scores[split:], window_size)
    pr_auc_df["window_start"] += split
    pr_auc_df["window_end"] += split

    logger.info(
        "adwin_online: done, %d drifts detected, %d retrains, total_train_time=%.2fs, mean PR-AUC over eval windows=%.4f",
        len(drift_points), n_retrains, total_train_time, pr_auc_df["pr_auc"].dropna().mean(),
    )

    return {
        "scores": scores,
        "pr_auc_over_time": pr_auc_df,
        "drift_points": drift_points,
        "metadata": {
            "method": "adwin_online",
            "train_rows": split,
            "eval_rows": n - split,
            "delta": delta,
            "n_drifts_detected": len(drift_points),
            "n_retrains": n_retrains,
            "total_boosting_rounds": total_rounds,
            "train_time_sec": total_train_time,
            "params": params,
        },
    }


def params_for_warm_start(params: dict) -> dict:
    """xgb.train's Learning API expects `eta`, not sklearn's `learning_rate`/`n_estimators` keys."""
    p = {k: v for k, v in params.items() if k not in ("n_estimators", "learning_rate", "eval_metric")}
    p["eta"] = params.get("learning_rate", 0.1)
    p["objective"] = "binary:logistic"
    p["base_score"] = 0.5
    return p
