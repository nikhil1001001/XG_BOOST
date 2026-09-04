"""Phase 6 stress tests: deliberately misconfigured hazard rate, and gradual (non-changepoint) drift.

Not one of Section 4's three named experiment scripts, but Phase 6 needs a home for two required,
distinct stress tests that don't fit the synthetic-benchmark or ablations entry points (both of which
already have a settled, documented purpose) -- a new, clearly-scoped file is the least-disruptive fit,
consistent with how Phase 5 added `elliptic_loader.py`/`lending_club_loader.py` under Section 6a's
"etc." rather than overloading an existing module.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ragb.baselines.adwin_online import run_adwin_online
from ragb.baselines.static_xgb import run_static_xgb
from ragb.bocpd.detector import run_bocpd_on_signal
from ragb.bocpd.signals import binary_log_loss, rolling_mean
from ragb.boosting.single_expert import run_single_expert
from ragb.data.synthetic_generator import SyntheticRegimeGenerator, SyntheticStreamConfig, generate_gradual_drift_stream
from ragb.eval.metrics import detect_events_from_probability_trace, detection_lag_and_false_alarms
from ragb.utils.logging_config import get_logger, setup_logging

REPO_ROOT = Path(__file__).resolve().parents[3]


def stress_test_hazard_misconfiguration(stream, args, logger) -> pd.DataFrame:
    """Confirms the expected, documented failure modes (Section 9) actually appear: a too-sensitive
    (high) hazard rate should produce a high false-alarm rate; a too-insensitive (low) hazard rate
    should miss real breaks / detect them with much higher lag.
    """
    split = int(len(stream.X) * args.initial_train_frac)
    static_result = run_static_xgb(stream.X, stream.y, initial_train_frac=args.initial_train_frac, seed=args.seed, window_size=args.window_size)
    y_eval = stream.y.iloc[split:].to_numpy()
    residual = binary_log_loss(y_eval, static_result["scores"][split:])
    smoothed = rolling_mean(residual, args.smoothing_window)
    true_breaks_eval = np.array([bp - split for bp in stream.breakpoints if bp > split])

    configs = [
        ("too_sensitive", args.oversensitive_hazard_rate),
        ("well_tuned", args.well_tuned_hazard_rate),
        ("too_insensitive", args.insensitive_hazard_rate),
    ]
    rows = []
    for label, hz in configs:
        bocpd_result = run_bocpd_on_signal(smoothed, hazard_rate=hz, break_window=args.break_window)
        events = detect_events_from_probability_trace(bocpd_result["post_break_weight"], threshold=args.event_threshold)
        m = detection_lag_and_false_alarms(true_breaks_eval, events, max_lag=args.max_lag, n_timesteps=len(smoothed))
        m["config"] = label
        m["hazard_rate"] = hz
        rows.append(m)
        logger.info(
            "hazard_misconfig: config=%s hazard_rate=%.6f detected=%d/%d false_alarms=%d (rate/1000=%.3f) mean_lag=%s",
            label, hz, m["n_detected"], m["n_true_breaks"], m["n_false_alarms"], m["false_alarm_rate_per_1000"], m["mean_lag"],
        )
    return pd.DataFrame(rows)[["config", "hazard_rate", "n_true_breaks", "n_detected", "n_missed", "mean_lag", "n_false_alarms", "false_alarm_rate_per_1000"]]


def stress_test_gradual_drift(args, logger) -> pd.DataFrame:
    """Injects gradual, non-changepoint drift (Section 9: "a known weak spot of changepoint-based
    methods generally... slow drift is ADWIN's territory, not RAGB's") and confirms RAGB
    (single_expert) underperforms the ADWIN baseline there -- an expected, honestly-reported
    limitation, not a bug to hide.
    """
    stream = generate_gradual_drift_stream(
        n_samples=args.gradual_n_samples, n_features=args.n_features,
        transition_length=args.transition_length, seed=args.seed,
    )
    logger.info(
        "gradual_drift: generated stream n_samples=%d transition_length=%d (no discrete breakpoint by construction)",
        args.gradual_n_samples, args.transition_length,
    )

    r_static = run_static_xgb(stream.X, stream.y, initial_train_frac=args.initial_train_frac, seed=args.seed, window_size=args.window_size)
    r_single = run_single_expert(
        stream.X, stream.y, initial_train_frac=args.initial_train_frac, chunk_size=args.chunk_size,
        hazard_rate=args.well_tuned_hazard_rate, boost_rounds_per_chunk=args.boost_rounds_per_chunk,
        seed=args.seed, window_size=args.window_size,
    )
    r_adwin = run_adwin_online(stream.X, stream.y, initial_train_frac=args.initial_train_frac, seed=args.seed, window_size=args.window_size)

    rows = [
        {"method": "static_xgb", "mean_pr_auc": r_static["pr_auc_over_time"]["pr_auc"].dropna().mean(), "total_boosting_rounds": r_static["metadata"]["total_boosting_rounds"]},
        {"method": "single_expert_RAGB", "mean_pr_auc": r_single["pr_auc_over_time"]["pr_auc"].dropna().mean(), "total_boosting_rounds": r_single["metadata"]["total_boosting_rounds"]},
        {"method": "adwin_online", "mean_pr_auc": r_adwin["pr_auc_over_time"]["pr_auc"].dropna().mean(), "total_boosting_rounds": r_adwin["metadata"]["total_boosting_rounds"]},
    ]
    for r in rows:
        logger.info("gradual_drift: method=%s mean_pr_auc=%.4f total_boosting_rounds=%d", r["method"], r["mean_pr_auc"], r["total_boosting_rounds"])
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6 stress tests: hazard misconfiguration + gradual drift")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=20_000)
    p.add_argument("--n-features", type=int, default=10)
    p.add_argument("--n-eras", type=int, default=3)
    p.add_argument("--generator-hazard-rate", type=float, default=1.0 / 1500.0)
    p.add_argument("--min-run-length", type=int, default=800)
    p.add_argument("--initial-train-frac", type=float, default=0.2)
    p.add_argument("--window-size", type=int, default=500)
    p.add_argument("--smoothing-window", type=int, default=20)
    p.add_argument("--break-window", type=int, default=15)
    p.add_argument("--event-threshold", type=float, default=0.3)
    p.add_argument("--max-lag", type=int, default=300)
    p.add_argument("--oversensitive-hazard-rate", type=float, default=0.05)
    p.add_argument("--well-tuned-hazard-rate", type=float, default=0.0004)
    p.add_argument("--insensitive-hazard-rate", type=float, default=0.00002)
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--boost-rounds-per-chunk", type=int, default=20)
    p.add_argument("--gradual-n-samples", type=int, default=8000)
    p.add_argument("--transition-length", type=int, default=3000)
    p.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "results" / "tables"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_path = setup_logging("phase6_stress_tests")
    logger = get_logger(__name__)

    run_start = time.time()
    logger.info("=== Phase 6 stress tests starting ===")
    logger.info("Resolved config: %s", vars(args))
    logger.info("Log file: %s", log_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("--- Stress test 1: hazard-rate misconfiguration ---")
    stream_cfg = SyntheticStreamConfig(
        n_samples=args.n_samples, n_features=args.n_features, n_eras=args.n_eras,
        hazard_rate=args.generator_hazard_rate, min_run_length=args.min_run_length, seed=args.seed,
    )
    stream = SyntheticRegimeGenerator(stream_cfg).generate()
    df1 = stress_test_hazard_misconfiguration(stream, args, logger)
    df1.to_csv(output_dir / "phase6_stress_hazard_misconfiguration.csv", index=False)
    logger.info("\n%s", df1.to_string(index=False))

    logger.info("--- Stress test 2: gradual (non-changepoint) drift ---")
    df2 = stress_test_gradual_drift(args, logger)
    df2.to_csv(output_dir / "phase6_stress_gradual_drift.csv", index=False)
    logger.info("\n%s", df2.to_string(index=False))

    duration = time.time() - run_start
    logger.info("=== Phase 6 stress tests complete in %.2fs ===", duration)


if __name__ == "__main__":
    main()
