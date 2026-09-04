import numpy as np
import pandas as pd

from ragb.bocpd.signals import binary_log_loss, feature_drift_kl_signal, rolling_mean


def test_binary_log_loss_zero_for_perfect_confident_prediction():
    y = np.array([1.0, 0.0])
    p = np.array([1.0 - 1e-9, 1e-9])
    loss = binary_log_loss(y, p)
    assert np.all(loss < 1e-6)


def test_binary_log_loss_high_for_confidently_wrong_prediction():
    y = np.array([1.0])
    p = np.array([1e-9])
    loss = binary_log_loss(y, p)
    assert loss[0] > 10


def test_rolling_mean_matches_manual_computation_at_full_window():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = rolling_mean(x, window=3)
    # once the window is fully "inside" the array, matches a plain trailing mean
    assert np.isclose(result[2], np.mean([1, 2, 3]))
    assert np.isclose(result[5], np.mean([4, 5, 6]))


def test_rolling_mean_shrinks_window_at_start():
    x = np.array([10.0, 20.0, 30.0])
    result = rolling_mean(x, window=5)
    assert np.isclose(result[0], 10.0)  # only 1 value available
    assert np.isclose(result[1], 15.0)  # mean of first 2
    assert np.isclose(result[2], 20.0)  # mean of all 3


def test_feature_drift_kl_signal_first_chunk_is_zero():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f0": rng.normal(0, 1, 1000), "f1": rng.normal(0, 1, 1000)})
    signal = feature_drift_kl_signal(X, chunk_size=100)
    assert signal[0] == 0.0


def test_feature_drift_kl_signal_spikes_at_a_real_distribution_shift():
    rng = np.random.default_rng(1)
    # 5 stable chunks, then a chunk with a shifted mean, then 4 more stable chunks at the new mean.
    chunks = [rng.normal(0, 1, 200) for _ in range(5)]
    chunks.append(rng.normal(8, 1, 200))  # sharp mean shift
    chunks += [rng.normal(8, 1, 200) for _ in range(4)]
    X = pd.DataFrame({"f0": np.concatenate(chunks)})
    signal = feature_drift_kl_signal(X, chunk_size=200)

    assert len(signal) == 10
    shift_chunk_idx = 5
    assert signal[shift_chunk_idx] > 5 * np.mean(signal[1:shift_chunk_idx])  # sharp spike vs. stable baseline
    assert signal[shift_chunk_idx + 1] < signal[shift_chunk_idx]  # settles back down once both chunks are post-shift


def test_feature_drift_kl_signal_raises_with_too_few_chunks():
    X = pd.DataFrame({"f0": np.arange(50)})
    try:
        feature_drift_kl_signal(X, chunk_size=100)
        assert False, "expected ValueError"
    except ValueError:
        pass
