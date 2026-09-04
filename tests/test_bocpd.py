import numpy as np

from ragb.bocpd.detector import BOCPD, run_bocpd_on_signal
from ragb.eval.metrics import detect_events_from_probability_trace, detection_lag_and_false_alarms


def _gaussian_mean_shift_signal(n=1000, changepoint=500, mean0=0.0, mean1=5.0, std=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.concatenate([
        rng.normal(mean0, std, size=changepoint),
        rng.normal(mean1, std, size=n - changepoint),
    ])
    return x


def test_run_length_posterior_sums_to_one_at_every_step():
    x = _gaussian_mean_shift_signal()
    detector = BOCPD(hazard_rate=1 / 250)
    for xi in x[:200]:
        posterior = detector.update(float(xi))
        assert np.isclose(posterior.sum(), 1.0, atol=1e-6)


def test_map_run_length_resets_near_known_changepoint():
    changepoint = 500
    x = _gaussian_mean_shift_signal(n=1000, changepoint=changepoint, mean0=0.0, mean1=6.0, std=1.0)
    result = run_bocpd_on_signal(x, hazard_rate=1 / 250)
    map_run_length = result["map_run_length"]

    # Right before the break, the model should have accumulated a long run (high MAP run length).
    assert map_run_length[changepoint - 5] > 20
    # Within a small tolerance window after the true break, the MAP run length should have collapsed
    # back down close to 0 (a changepoint was detected), not kept growing as if nothing happened.
    post_break_window = map_run_length[changepoint:changepoint + 20]
    assert post_break_window.min() <= 5


def test_changepoint_probability_is_pinned_to_hazard_rate_constant():
    """Documents a real mathematical property of constant-hazard BOCPD (not a bug): P(r_t=0|x_1:t)
    is provably identical to the hazard rate at every timestep regardless of the data, because its
    joint probability is always exactly `hazard_rate * evidence`, and the evidence term is also the
    total normalizer. This is why detection uses `post_break_weight` instead (next test).
    """
    hazard_rate = 1 / 250
    x = _gaussian_mean_shift_signal(n=1000, changepoint=500, mean0=0.0, mean1=6.0, std=1.0)
    result = run_bocpd_on_signal(x, hazard_rate=hazard_rate)
    np.testing.assert_allclose(result["changepoint_prob"], hazard_rate, atol=1e-9)


def test_post_break_weight_spikes_at_known_location_within_tolerance():
    changepoint = 500
    tolerance = 15
    x = _gaussian_mean_shift_signal(n=1000, changepoint=changepoint, mean0=0.0, mean1=6.0, std=1.0)
    result = run_bocpd_on_signal(x, hazard_rate=1 / 250, break_window=10)
    pbw = result["post_break_weight"]

    peak_idx = int(np.argmax(pbw[changepoint - tolerance:changepoint + 50])) + (changepoint - tolerance)
    assert abs(peak_idx - changepoint) <= tolerance
    assert pbw[peak_idx] > 0.3  # a real spike, not noise-level
    assert pbw[peak_idx] > 5 * pbw[changepoint - 100:changepoint - 10].mean()  # well above pre-break baseline


def test_detection_lag_and_false_alarms_on_known_changepoint():
    changepoint = 500
    x = _gaussian_mean_shift_signal(n=1000, changepoint=changepoint, mean0=0.0, mean1=6.0, std=1.0)
    result = run_bocpd_on_signal(x, hazard_rate=1 / 250, break_window=10)
    events = detect_events_from_probability_trace(result["post_break_weight"], threshold=0.3)
    metrics = detection_lag_and_false_alarms(
        true_breakpoints=np.array([changepoint]), detected_events=events, max_lag=30, n_timesteps=len(x),
    )
    assert metrics["n_detected"] == 1
    assert metrics["n_missed"] == 0
    assert metrics["mean_lag"] <= 30


def test_post_break_weight_high_right_after_reset_low_mid_run():
    x = _gaussian_mean_shift_signal(n=800, changepoint=400, mean0=0.0, mean1=6.0, std=1.0)
    detector = BOCPD(hazard_rate=1 / 250)
    weights = []
    for xi in x:
        detector.update(float(xi))
        weights.append(detector.post_break_weight(break_window=10))
    weights = np.array(weights)
    # Right after the true break, P(recently reset) should be much higher than deep into a stable run.
    assert weights[405:415].mean() > weights[300:390].mean()


def test_no_changepoint_low_false_alarm_rate_on_stationary_signal():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, size=1000)
    result = run_bocpd_on_signal(x, hazard_rate=1 / 250, break_window=10)
    events = detect_events_from_probability_trace(result["post_break_weight"], threshold=0.5)
    # A stationary signal shouldn't spuriously fire many high-confidence changepoint events.
    assert len(events) <= 5
