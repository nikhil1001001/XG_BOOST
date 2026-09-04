"""Soft-weighted gradient/Hessian obj(preds, dtrain) callback for XGBoost: G = sum(w_i*g_i), H = sum(w_i*h_i).

Implements the "Math core" of Section 3 point 4 exactly: standard binary logloss gradient/Hessian
per instance is grad_i = p_i - y_i, hess_i = p_i*(1-p_i) (p_i = sigmoid(raw_pred_i)). XGBoost builds
each leaf's weight as -G/(H+lambda) where G, H are SUMS of grad/hess over the instances landing in
that leaf -- so scaling each instance's own grad_i, hess_i by its soft weight w_i before XGBoost sums
them is exactly equivalent to the brief's G = sum(w_i*g_i), H = sum(w_i*h_i) formula. This is done via
a custom obj() callback (not XGBoost's built-in per-instance `weight` field on DMatrix) so the weight
source is explicit and auditable as the BOCPD-derived soft weight, not conflated with any other use of
instance weighting.
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb


def make_soft_weighted_logloss_obj(weights: np.ndarray):
    """Returns an obj(preds, dtrain) closure for XGBoost's `obj=` training argument.

    `weights` must be aligned index-for-index with the DMatrix's instance order and is captured by
    the closure -- weights don't change mid-training-call, matching Section 3's per-instance
    "posterior probability of belonging to the current regime" being computed once (from BOCPD state)
    before a sub-ensemble's training round(s) start, not re-derived every boosting iteration.
    """
    weights = np.asarray(weights, dtype=np.float64)

    def obj(preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        y = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-preds))
        grad = (p - y) * weights
        hess = np.maximum(p * (1.0 - p) * weights, 1e-16)  # numerical floor, negligible vs. any real lambda
        return grad, hess

    return obj
