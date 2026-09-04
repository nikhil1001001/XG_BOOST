import numpy as np

from ragb.boosting.mixture import (
    MixtureExpert,
    _blend_weights,
    decide_promotion,
    run_mixture,
    select_expert_to_prune,
)
from ragb.data.synthetic_generator import SyntheticRegimeGenerator, SyntheticStreamConfig


def _expert(id_, birth_time, status="active"):
    return MixtureExpert(id=id_, birth_time=birth_time, booster=None, status=status)


# ---- decide_promotion (pure function, constructed fixtures per Section 13) ----

def test_promotion_when_candidate_strictly_lowers_loss():
    promoted, mean_with, mean_without = decide_promotion([0.1, 0.2, 0.15], [0.3, 0.35, 0.32])
    assert promoted is True
    assert mean_with < mean_without


def test_discard_when_candidate_does_not_help():
    promoted, mean_with, mean_without = decide_promotion([0.4, 0.5], [0.2, 0.25])
    assert promoted is False


def test_discard_on_empty_loss_lists_defaults_to_infinite_loss_no_crash():
    promoted, mean_with, mean_without = decide_promotion([], [])
    assert promoted is False
    assert mean_with == float("inf")
    assert mean_without == float("inf")


# ---- select_expert_to_prune (pure function, constructed fixtures) ----

def test_no_pruning_when_at_or_under_capacity():
    experts = [_expert(0, 0), _expert(1, 100), _expert(2, 200)]
    assert select_expert_to_prune(experts, K=3) is None
    assert select_expert_to_prune(experts, K=5) is None


def test_prunes_oldest_expert_when_over_capacity():
    experts = [_expert(0, 0), _expert(1, 100), _expert(2, 200), _expert(3, 300)]
    pruned = select_expert_to_prune(experts, K=3)
    assert pruned.id == 0
    assert pruned.birth_time == 0


# ---- _blend_weights (pure function, constructed posterior fixtures) ----

def test_blend_weights_single_expert_gets_full_weight():
    experts = [_expert(0, 0)]
    posterior = np.array([0.2, 0.3, 0.5])
    weights = _blend_weights(experts, now=10, run_length_posterior=posterior)
    assert weights == {0: 1.0}


def test_blend_weights_two_experts_partition_matches_hand_calculation():
    # now=250, older expert born at 100, newer at 200 -> newer's bin is r in [0,50], older's is r in [51, inf)
    experts = [_expert(0, 100), _expert(1, 200)]
    posterior = np.zeros(300)
    posterior[30] = 0.4  # r=30 -> falls in newer expert's bin (born 200, now-30=220 >= 200)
    posterior[100] = 0.6  # r=100 -> falls in older expert's bin (now-100=150, between 100 and 200)
    weights = _blend_weights(experts, now=250, run_length_posterior=posterior)
    assert abs(weights[1] - 0.4) < 1e-9  # newer expert gets the r=30 mass
    assert abs(weights[0] - 0.6) < 1e-9  # older expert gets the r=100 mass
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_blend_weights_three_experts_sum_to_one():
    experts = [_expert(0, 0), _expert(1, 100), _expert(2, 200)]
    rng = np.random.default_rng(0)
    posterior = rng.random(260)
    posterior /= posterior.sum()
    weights = _blend_weights(experts, now=250, run_length_posterior=posterior)
    assert set(weights.keys()) == {0, 1, 2}
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert all(w >= 0 for w in weights.values())


def test_blend_weights_favors_newest_expert_when_posterior_concentrated_at_low_run_length():
    experts = [_expert(0, 0), _expert(1, 100), _expert(2, 200)]
    posterior = np.zeros(300)
    posterior[2] = 1.0  # near-certain the current run just started (r=2)
    weights = _blend_weights(experts, now=250, run_length_posterior=posterior)
    assert weights[2] > 0.99  # newest expert (born 200) should get essentially all the blend weight


# ---- end-to-end smoke test (real training, not just fixtures) ----

def test_run_mixture_end_to_end_runs_without_error():
    cfg = SyntheticStreamConfig(n_samples=6000, n_features=5, n_eras=3, hazard_rate=1 / 900, min_run_length=300, seed=11)
    stream = SyntheticRegimeGenerator(cfg).generate()
    result = run_mixture(
        stream.X, stream.y, initial_train_frac=0.25, chunk_size=300, probation_window=600,
        window_size=200, seed=11,
    )
    df = result["pr_auc_over_time"]
    assert len(df) > 0
    assert df["pr_auc"].notna().any()
    meta = result["metadata"]
    assert meta["n_promoted"] + meta["n_discarded"] == meta["n_spawns_total"]
    if meta["n_spawns_total"] > 0:
        assert not np.isnan(meta["spurious_spawn_rate"])
    assert meta["n_active_experts_final"] <= meta["K"]


def test_run_mixture_reproducible_given_same_seed():
    cfg = SyntheticStreamConfig(n_samples=4000, n_features=5, n_eras=2, hazard_rate=1 / 600, min_run_length=250, seed=5)
    stream = SyntheticRegimeGenerator(cfg).generate()
    r1 = run_mixture(stream.X, stream.y, initial_train_frac=0.25, chunk_size=300, window_size=200, seed=5)
    r2 = run_mixture(stream.X, stream.y, initial_train_frac=0.25, chunk_size=300, window_size=200, seed=5)
    np.testing.assert_allclose(
        r1["scores"][~np.isnan(r1["scores"])],
        r2["scores"][~np.isnan(r2["scores"])],
    )
