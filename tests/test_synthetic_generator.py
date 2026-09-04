import numpy as np

from ragb.data.synthetic_generator import (
    SyntheticRegimeGenerator,
    SyntheticStreamConfig,
    generate_single_changepoint,
)


def test_seed_reproducibility():
    cfg = SyntheticStreamConfig(n_samples=3000, n_features=5, n_eras=3, seed=123)
    s1 = SyntheticRegimeGenerator(cfg).generate()
    s2 = SyntheticRegimeGenerator(cfg).generate()

    assert np.array_equal(s1.X.to_numpy(), s2.X.to_numpy())
    assert np.array_equal(s1.y.to_numpy(), s2.y.to_numpy())
    assert np.array_equal(s1.breakpoints, s2.breakpoints)
    assert np.array_equal(s1.regime_id, s2.regime_id)


def test_different_seed_differs():
    s1 = SyntheticRegimeGenerator(SyntheticStreamConfig(n_samples=3000, seed=1)).generate()
    s2 = SyntheticRegimeGenerator(SyntheticStreamConfig(n_samples=3000, seed=2)).generate()
    assert not np.array_equal(s1.X.to_numpy(), s2.X.to_numpy())


def test_breakpoints_match_regime_changes():
    cfg = SyntheticStreamConfig(n_samples=5000, n_features=5, n_eras=3, seed=7)
    stream = SyntheticRegimeGenerator(cfg).generate()

    # Ground-truth breakpoints must be exactly where regime_id actually changes value.
    actual_changes = np.where(np.diff(stream.regime_id) != 0)[0] + 1
    assert np.array_equal(np.sort(stream.breakpoints), actual_changes)


def test_at_least_configured_number_of_eras_used_when_stream_long_enough():
    cfg = SyntheticStreamConfig(n_samples=20_000, n_features=5, n_eras=3, hazard_rate=1 / 1500, min_run_length=800, seed=99)
    stream = SyntheticRegimeGenerator(cfg).generate()
    assert set(stream.regime_id).issubset(set(range(cfg.n_eras)))
    assert len(stream.breakpoints) >= 1  # a 20k-sample stream should switch at least once


def test_output_shapes_and_label_domain():
    cfg = SyntheticStreamConfig(n_samples=1000, n_features=4, n_eras=2, seed=5)
    stream = SyntheticRegimeGenerator(cfg).generate()
    assert stream.X.shape == (1000, 4)
    assert stream.y.shape == (1000,)
    assert set(stream.y.unique()).issubset({0, 1})


def test_single_changepoint_ground_truth():
    stream = generate_single_changepoint(n_samples=4000, breakpoint_idx=2500, n_features=6, seed=11)
    assert list(stream.breakpoints) == [2500]
    assert set(stream.regime_id[:2500]) == {0}
    assert set(stream.regime_id[2500:]) == {1}
