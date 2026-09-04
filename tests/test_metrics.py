import numpy as np

from ragb.eval.metrics import (
    detect_events_from_probability_trace,
    detection_lag_and_false_alarms,
    pr_auc,
    windowed_pr_auc,
)


def test_pr_auc_nan_when_single_class():
    assert np.isnan(pr_auc(np.zeros(10), np.random.rand(10)))


def test_pr_auc_perfect_separation_is_one():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    assert pr_auc(y, scores) == 1.0


def test_windowed_pr_auc_covers_all_rows_and_windows():
    y = np.random.default_rng(0).integers(0, 2, size=1000)
    scores = np.random.default_rng(1).random(1000)
    df = windowed_pr_auc(y, scores, window_size=300)
    assert df["n"].sum() == 1000
    assert len(df) == 4  # 300, 300, 300, 100
    assert df.iloc[-1]["n"] == 100


def test_detect_events_collapses_contiguous_runs():
    trace = np.array([0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.8, 0.0])
    events = detect_events_from_probability_trace(trace, threshold=0.5)
    assert list(events) == [1, 6]


def test_detect_events_empty_when_never_above_threshold():
    trace = np.zeros(20)
    events = detect_events_from_probability_trace(trace, threshold=0.5)
    assert len(events) == 0


def test_detection_lag_matches_nearest_unused_detection():
    true_breaks = np.array([100, 300])
    detected = np.array([105, 110, 305])  # two detections near the first break, one near the second
    m = detection_lag_and_false_alarms(true_breaks, detected, max_lag=20, n_timesteps=1000)
    assert m["n_detected"] == 2
    assert m["n_missed"] == 0
    assert m["n_false_alarms"] == 1  # the second detection near break 1 (110) is unmatched
    assert m["mean_lag"] == np.mean([5, 5])  # nearest match to break 1 is 105 (lag 5), break 2 is 305 (lag 5)


def test_detection_lag_reports_missed_break_when_no_detection_in_window():
    true_breaks = np.array([100])
    detected = np.array([500])  # far outside max_lag
    m = detection_lag_and_false_alarms(true_breaks, detected, max_lag=20, n_timesteps=1000)
    assert m["n_detected"] == 0
    assert m["n_missed"] == 1
    assert m["n_false_alarms"] == 1
    assert np.isnan(m["mean_lag"])
