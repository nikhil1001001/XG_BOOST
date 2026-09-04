import numpy as np

from ragb.boosting.single_expert import survival_weights, run_single_expert
from ragb.data.synthetic_generator import SyntheticRegimeGenerator, SyntheticStreamConfig

_SMALL_CFG = SyntheticStreamConfig(n_samples=3000, n_features=5, n_eras=2, hazard_rate=1 / 400, min_run_length=200, seed=3)


def test_survival_weights_recent_instance_gets_near_full_weight():
    # A posterior with all mass at run length r=50 (P(r=50)=1) means the current run is known to
    # extend back exactly 50 steps: any instance at distance <= 50 is definitely within that run
    # (weight 1), any instance further back than that is definitely from an earlier, ended run
    # (weight 0).
    posterior = np.zeros(60)
    posterior[50] = 1.0
    weights = survival_weights(posterior, distances=np.array([0, 2, 49, 50, 55]))
    assert weights[0] == 1.0
    assert weights[1] == 1.0
    assert weights[2] == 1.0
    assert weights[3] == 1.0  # distance == r exactly: still within the known run
    assert weights[4] == 0.0  # beyond the run length entirely


def test_survival_weights_beyond_posterior_length_is_zero():
    posterior = np.array([0.5, 0.3, 0.2])
    weights = survival_weights(posterior, distances=np.array([0, 1, 2, 10, 100]))
    assert weights[3] == 0.0
    assert weights[4] == 0.0


def test_survival_weights_monotonically_non_increasing_with_distance():
    rng = np.random.default_rng(0)
    posterior = rng.random(30)
    posterior /= posterior.sum()
    distances = np.arange(30)
    weights = survival_weights(posterior, distances)
    assert np.all(np.diff(weights) <= 1e-12)  # non-increasing (survival function property)


def test_run_single_expert_runs_without_error_and_produces_pr_auc():
    stream = SyntheticRegimeGenerator(_SMALL_CFG).generate()
    result = run_single_expert(
        stream.X, stream.y, initial_train_frac=0.3, chunk_size=300, window_size=200, seed=3,
    )
    df = result["pr_auc_over_time"]
    assert len(df) > 0
    assert df["pr_auc"].notna().any()
    assert result["metadata"]["n_chunks"] > 1
    assert result["metadata"]["total_boosting_rounds"] > result["metadata"]["boost_rounds_per_chunk"]


def test_run_single_expert_reproducible_given_same_seed():
    stream = SyntheticRegimeGenerator(_SMALL_CFG).generate()
    r1 = run_single_expert(stream.X, stream.y, initial_train_frac=0.3, chunk_size=300, window_size=200, seed=3)
    r2 = run_single_expert(stream.X, stream.y, initial_train_frac=0.3, chunk_size=300, window_size=200, seed=3)
    np.testing.assert_allclose(
        r1["scores"][~np.isnan(r1["scores"])],
        r2["scores"][~np.isnan(r2["scores"])],
    )


def test_weights_are_bounded_between_zero_and_one():
    stream = SyntheticRegimeGenerator(_SMALL_CFG).generate()
    result = run_single_expert(stream.X, stream.y, initial_train_frac=0.3, chunk_size=300, window_size=200, seed=3)
    ws = result["weight_stats"]
    assert (ws["weight_min"] >= 0).all()
    assert (ws["weight_max"] <= 1.0 + 1e-9).all()
