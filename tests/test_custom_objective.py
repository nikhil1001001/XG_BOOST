import numpy as np
import xgboost as xgb

from ragb.boosting.custom_objective import make_soft_weighted_logloss_obj


def test_grad_hess_formula_matches_closed_form_binary_logloss():
    rng = np.random.default_rng(0)
    preds = rng.normal(0, 2, size=20)  # raw margins
    y = rng.integers(0, 2, size=20).astype(float)

    dtrain = xgb.DMatrix(rng.normal(size=(20, 3)), label=y)
    obj = make_soft_weighted_logloss_obj(weights=np.ones(20))
    grad, hess = obj(preds, dtrain)

    p = 1.0 / (1.0 + np.exp(-preds))
    expected_grad = p - y
    expected_hess = p * (1 - p)

    np.testing.assert_allclose(grad, expected_grad, atol=1e-12)
    np.testing.assert_allclose(hess, expected_hess, atol=1e-12)


def test_unit_weights_scale_grad_hess_linearly():
    rng = np.random.default_rng(1)
    preds = rng.normal(0, 1, size=15)
    y = rng.integers(0, 2, size=15).astype(float)
    weights = rng.uniform(0.1, 1.0, size=15)

    dtrain = xgb.DMatrix(rng.normal(size=(15, 2)), label=y)
    obj_weighted = make_soft_weighted_logloss_obj(weights=weights)
    obj_unit = make_soft_weighted_logloss_obj(weights=np.ones(15))

    grad_w, hess_w = obj_weighted(preds, dtrain)
    grad_u, hess_u = obj_unit(preds, dtrain)

    np.testing.assert_allclose(grad_w, grad_u * weights, atol=1e-12)
    np.testing.assert_allclose(hess_w, hess_u * weights, atol=1e-12)


def test_reduces_to_standard_boosting_when_all_weights_are_one():
    """Acceptance criterion (Section 5, Phase 3): training with the custom soft-weighted obj() at
    weights=1 must reduce EXACTLY to standard XGBoost training with the built-in binary:logistic
    objective -- verified end-to-end by training two boosters with identical data/params/rounds
    (one via the built-in objective, one via our custom obj with unit weights) and checking their
    raw-margin predictions match to numerical precision.
    """
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 5))
    true_w = rng.normal(size=5)
    y = (rng.normal(size=n) + X @ true_w > 0).astype(float)

    dtrain = xgb.DMatrix(X, label=y)
    base_params = dict(max_depth=3, eta=0.3, tree_method="hist", base_score=0.5, objective="binary:logistic")

    booster_builtin = xgb.train(base_params, dtrain, num_boost_round=10)

    obj = make_soft_weighted_logloss_obj(weights=np.ones(n))
    booster_custom = xgb.train(base_params, dtrain, num_boost_round=10, obj=obj)

    preds_builtin = booster_builtin.predict(dtrain, output_margin=True)
    preds_custom = booster_custom.predict(dtrain, output_margin=True)

    np.testing.assert_allclose(preds_builtin, preds_custom, atol=1e-5)


def test_zero_weight_instances_contribute_nothing():
    rng = np.random.default_rng(2)
    preds = rng.normal(0, 1, size=10)
    y = rng.integers(0, 2, size=10).astype(float)
    weights = np.zeros(10)

    dtrain = xgb.DMatrix(rng.normal(size=(10, 2)), label=y)
    obj = make_soft_weighted_logloss_obj(weights=weights)
    grad, hess = obj(preds, dtrain)

    np.testing.assert_allclose(grad, np.zeros(10), atol=1e-12)
    assert np.all(hess > 0)  # numerical floor keeps hess strictly positive, but negligible
    assert np.all(hess < 1e-10)
