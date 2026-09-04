import numpy as np

from ragb.baselines.periodic_retrain import run_periodic_retrain
from ragb.baselines.static_xgb import run_static_xgb
from ragb.data.synthetic_generator import SyntheticRegimeGenerator, SyntheticStreamConfig

# Small stream so this stays well under the "don't tqdm-wrap trivial/fast loops" ~1s threshold
# implied for tests, while still exercising the real training/prediction path end to end.
_SMALL_CFG = SyntheticStreamConfig(n_samples=2000, n_features=5, n_eras=2, hazard_rate=1 / 500, min_run_length=200, seed=1)


def _small_stream():
    return SyntheticRegimeGenerator(_SMALL_CFG).generate()


def test_static_xgb_runs_without_error_and_produces_pr_auc():
    stream = _small_stream()
    result = run_static_xgb(stream.X, stream.y, initial_train_frac=0.3, window_size=200, seed=1)
    assert "pr_auc_over_time" in result
    df = result["pr_auc_over_time"]
    assert len(df) > 0
    assert df["pr_auc"].notna().any()
    assert result["metadata"]["n_retrains"] == 1


def test_periodic_retrain_runs_without_error_and_produces_pr_auc():
    stream = _small_stream()
    result = run_periodic_retrain(stream.X, stream.y, initial_train_frac=0.3, retrain_every=400, window_size=200, seed=1)
    df = result["pr_auc_over_time"]
    assert len(df) > 0
    assert df["pr_auc"].notna().any()
    assert result["metadata"]["n_retrains"] > 1


def test_baselines_reproducible_given_same_seed():
    stream = _small_stream()
    r1 = run_static_xgb(stream.X, stream.y, initial_train_frac=0.3, window_size=200, seed=1)
    r2 = run_static_xgb(stream.X, stream.y, initial_train_frac=0.3, window_size=200, seed=1)
    np.testing.assert_allclose(
        r1["scores"][~np.isnan(r1["scores"])],
        r2["scores"][~np.isnan(r2["scores"])],
    )
