"""Entry point: run the full RAGB pipeline + all baselines on the synthetic regime-switching benchmark.

Phase 1 version runs the generator plus the two Phase-1 baselines (static_xgb, periodic_retrain).
Later phases extend this same script to add BOCPD-aware baselines and RAGB itself rather than
introducing a parallel entry point, so results across phases stay in one comparable table.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ragb.baselines.adwin_online import run_adwin_online
from ragb.baselines.periodic_retrain import run_periodic_retrain
from ragb.baselines.sliding_window_retrain import run_sliding_window_retrain
from ragb.baselines.static_xgb import run_static_xgb
from ragb.bocpd.detector import run_bocpd_on_signal
from ragb.bocpd.signals import binary_log_loss, rolling_mean
from ragb.boosting.mixture import run_mixture
from ragb.boosting.single_expert import run_single_expert
from ragb.data.synthetic_generator import SyntheticRegimeGenerator, SyntheticStream, SyntheticStreamConfig
from ragb.eval.metrics import detect_events_from_probability_trace, detection_lag_and_false_alarms
from ragb.utils.logging_config import get_logger, setup_logging

REPO_ROOT = Path(__file__).resolve().parents[3]
logger = get_logger(__name__)


def run_bocpd_validation(
    stream: SyntheticStream,
    static_scores: np.ndarray,
    split: int,
    hazard_rates: list[float],
    smoothing_window: int = 20,
    break_window: int = 15,
    event_threshold: float = 0.3,
    max_lag: int = 300,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Phase 2: validates BOCPD on the residual-loss signal (Section 3 point 1) computed from the
    already-trained static_xgb classifier's predictions past its training cutoff, against the
    synthetic generator's known ground-truth breakpoints. Sweeps hazard_rate to characterize the
    detection-lag / false-alarm-rate tradeoff (Section 8, Phase 2 acceptance criteria) rather than
    reporting one cherry-picked operating point.
    """
    y_eval = stream.y.iloc[split:].to_numpy()
    scores_eval = static_scores[split:]
    residual_loss = binary_log_loss(y_eval, scores_eval)
    smoothed = rolling_mean(residual_loss, smoothing_window)

    # Only breakpoints after the classifier's training cutoff are detectable from this signal (the
    # classifier has no residual-loss signal to react to before it starts scoring).
    true_breaks_eval = np.array([bp - split for bp in stream.breakpoints if bp > split])
    logger.info(
        "BOCPD validation: %d evaluable breakpoints (of %d total), smoothing_window=%d, break_window=%d, event_threshold=%.2f",
        len(true_breaks_eval), len(stream.breakpoints), smoothing_window, break_window, event_threshold,
    )

    rows = []
    for hazard_rate in hazard_rates:
        result = run_bocpd_on_signal(smoothed, hazard_rate=hazard_rate, break_window=break_window)
        events = detect_events_from_probability_trace(result["post_break_weight"], threshold=event_threshold)
        m = detection_lag_and_false_alarms(true_breaks_eval, events, max_lag=max_lag, n_timesteps=len(smoothed))
        m["hazard_rate"] = hazard_rate
        rows.append(m)
        logger.info(
            "hazard_rate=%.5f: detected=%d/%d missed=%d mean_lag=%.1f false_alarms=%d (rate/1000=%.3f)",
            hazard_rate, m["n_detected"], m["n_true_breaks"], m["n_missed"],
            m["mean_lag"], m["n_false_alarms"], m["false_alarm_rate_per_1000"],
        )

    sweep_df = pd.DataFrame(rows)[[
        "hazard_rate", "n_true_breaks", "n_detected", "n_missed",
        "mean_lag", "n_false_alarms", "false_alarm_rate_per_1000",
    ]]

    if output_dir is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(sweep_df["hazard_rate"], sweep_df["mean_lag"], marker="o")
        axes[0].set_xscale("log")
        axes[0].set_xlabel("hazard_rate")
        axes[0].set_ylabel("mean detection lag (timesteps)")
        axes[0].set_title("Detection lag vs. hazard rate")

        axes[1].plot(sweep_df["hazard_rate"], sweep_df["false_alarm_rate_per_1000"], marker="o", color="tab:red")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("hazard_rate")
        axes[1].set_ylabel("false alarms per 1000 steps")
        axes[1].set_title("False-alarm rate vs. hazard rate")

        fig.tight_layout()
        fig_path = Path(output_dir) / "phase2_hazard_sweep.png"
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        logger.info("Wrote hazard-sweep figure: %s", fig_path)

    return sweep_df


def _rewrite_combined_tables(results: dict, combined_path: Path, summary_path: Path, seed: int) -> None:
    """Rewrites the combined windowed-PR-AUC and summary tables from whichever methods have run so
    far in `results`, so results/tables/phase1_baselines_* stays the one place all synthetic-benchmark
    methods are compared (per this script's stated design), regardless of which phase last added a
    method to the dict.
    """
    combined_rows = []
    for method, res in results.items():
        df = res["pr_auc_over_time"].copy()
        df["method"] = method
        combined_rows.append(df)
    combined = pd.concat(combined_rows, ignore_index=True)
    combined.to_csv(combined_path, index=False)
    logger.info("Updated windowed PR-AUC table: %s (%d rows, methods=%s)", combined_path, len(combined), list(results.keys()))

    summary_rows = []
    for method, res in results.items():
        meta = res["metadata"]
        summary_rows.append({
            "method": method,
            "mean_pr_auc": res["pr_auc_over_time"]["pr_auc"].dropna().mean(),
            "n_retrains": meta["n_retrains"],
            "total_boosting_rounds": meta["total_boosting_rounds"],
            "train_time_sec": meta["train_time_sec"],
            "seed": seed,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    logger.info("Updated summary table: %s\n%s", summary_path, summary_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthetic benchmark: Phase 1 baselines (static_xgb, periodic_retrain) "
                     "+ Phase 2 BOCPD hazard-rate sweep validation"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=20_000)
    p.add_argument("--n-features", type=int, default=10)
    p.add_argument("--n-eras", type=int, default=3)
    p.add_argument("--hazard-rate", type=float, default=1.0 / 1500.0, help="generator's own switch hazard rate")
    p.add_argument("--min-run-length", type=int, default=800)
    p.add_argument("--initial-train-frac", type=float, default=0.2)
    p.add_argument("--retrain-every", type=int, default=1000)
    p.add_argument("--window-size", type=int, default=500)
    p.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "results" / "tables"))
    p.add_argument(
        "--bocpd-hazard-sweep", type=float, nargs="+",
        default=[1 / 100, 1 / 250, 1 / 500, 1 / 1000, 1 / 2500, 1 / 5000],
        help="BOCPD detector hazard rates to sweep in Phase 2 validation (independent of the generator's own hazard rate)",
    )
    p.add_argument("--bocpd-smoothing-window", type=int, default=20)
    p.add_argument("--bocpd-break-window", type=int, default=15)
    p.add_argument("--bocpd-event-threshold", type=float, default=0.3)
    p.add_argument("--bocpd-max-lag", type=int, default=300)
    p.add_argument("--skip-bocpd", action="store_true", help="skip Phase 2 BOCPD validation, run Phase 1 baselines only")
    p.add_argument("--skip-single-expert", action="store_true", help="skip Phase 3 single-expert run")
    p.add_argument(
        "--single-expert-hazard-rate", type=float, default=0.0004,
        help="BOCPD hazard rate used by single_expert's soft-weighting. NOT the same tuning target as "
             "Phase 2's detection sweep -- Phase 3 found 0.002 (Phase 2's detection-recall pick) causes "
             "overly aggressive down-weighting when used for instance weights; a rate closer to the "
             "generator's own true switch rate (1/1500 here) works better for weighting. See phase3 report.",
    )
    p.add_argument("--single-expert-chunk-size", type=int, default=500)
    p.add_argument("--single-expert-boost-rounds-per-chunk", type=int, default=20)
    p.add_argument("--skip-mixture", action="store_true", help="skip Phase 4 mixture-of-experts run")
    p.add_argument("--mixture-K", type=int, default=3)
    p.add_argument("--mixture-spawn-threshold", type=float, default=0.3)
    p.add_argument("--mixture-probation-window", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_path = setup_logging("phase1_synthetic_baselines")

    run_start = time.time()
    logger.info("=== Phase 1 synthetic benchmark run starting ===")
    logger.info("Resolved config: %s", vars(args))
    logger.info("Log file: %s", log_path)

    stream_cfg = SyntheticStreamConfig(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_eras=args.n_eras,
        hazard_rate=args.hazard_rate,
        min_run_length=args.min_run_length,
        seed=args.seed,
    )
    stream = SyntheticRegimeGenerator(stream_cfg).generate()
    logger.info(
        "Data shape: X=%s y=%s, %d ground-truth breakpoints, fraud rate=%.4f",
        stream.X.shape, stream.y.shape, len(stream.breakpoints), stream.y.mean(),
    )

    results = {}
    results["static_xgb"] = run_static_xgb(
        stream.X, stream.y,
        initial_train_frac=args.initial_train_frac,
        seed=args.seed,
        window_size=args.window_size,
    )
    results["periodic_retrain"] = run_periodic_retrain(
        stream.X, stream.y,
        initial_train_frac=args.initial_train_frac,
        retrain_every=args.retrain_every,
        seed=args.seed,
        window_size=args.window_size,
    )
    results["sliding_window_retrain"] = run_sliding_window_retrain(
        stream.X, stream.y,
        initial_train_frac=args.initial_train_frac,
        retrain_every=args.retrain_every,
        seed=args.seed,
        window_size=args.window_size,
    )
    results["adwin_online"] = run_adwin_online(
        stream.X, stream.y,
        initial_train_frac=args.initial_train_frac,
        seed=args.seed,
        window_size=args.window_size,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "phase1_baselines_pr_auc_over_time.csv"
    summary_path = output_dir / "phase1_baselines_summary.csv"
    _rewrite_combined_tables(results, combined_path, summary_path, args.seed)

    breakpoints_path = output_dir / "phase1_ground_truth_breakpoints.json"
    with open(breakpoints_path, "w") as f:
        json.dump({
            "seed": args.seed,
            "n_samples": args.n_samples,
            "breakpoints": stream.breakpoints.tolist(),
            "config": vars(args),
        }, f, indent=2)
    logger.info("Wrote ground-truth breakpoints: %s", breakpoints_path)

    duration = time.time() - run_start
    logger.info("=== Phase 1 synthetic benchmark run complete in %.2fs ===", duration)

    if args.skip_bocpd:
        return

    bocpd_log_path = setup_logging("phase2_bocpd_hazard_sweep")
    bocpd_start = time.time()
    logger.info("=== Phase 2 BOCPD hazard-rate sweep validation starting ===")
    logger.info("Log file: %s", bocpd_log_path)
    logger.info(
        "Using residual-loss signal from the already-trained static_xgb classifier "
        "(same run as above, seed=%d, %d ground-truth breakpoints)", args.seed, len(stream.breakpoints),
    )

    split = int(args.n_samples * args.initial_train_frac)
    sweep_df = run_bocpd_validation(
        stream=stream,
        static_scores=results["static_xgb"]["scores"],
        split=split,
        hazard_rates=args.bocpd_hazard_sweep,
        smoothing_window=args.bocpd_smoothing_window,
        break_window=args.bocpd_break_window,
        event_threshold=args.bocpd_event_threshold,
        max_lag=args.bocpd_max_lag,
        output_dir=REPO_ROOT / "results" / "figures",
    )
    sweep_df["seed"] = args.seed
    sweep_path = output_dir / "phase2_bocpd_hazard_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    logger.info("Wrote BOCPD hazard-sweep table: %s\n%s", sweep_path, sweep_df.to_string(index=False))

    bocpd_duration = time.time() - bocpd_start
    logger.info("=== Phase 2 BOCPD hazard-rate sweep validation complete in %.2fs ===", bocpd_duration)

    if args.skip_single_expert:
        return

    se_log_path = setup_logging("phase3_single_expert")
    se_start = time.time()
    logger.info("=== Phase 3 single-expert soft-weighted pipeline starting ===")
    logger.info("Log file: %s", se_log_path)

    se_result = run_single_expert(
        stream.X, stream.y,
        initial_train_frac=args.initial_train_frac,
        chunk_size=args.single_expert_chunk_size,
        hazard_rate=args.single_expert_hazard_rate,
        boost_rounds_per_chunk=args.single_expert_boost_rounds_per_chunk,
        seed=args.seed,
        window_size=args.window_size,
    )
    results["single_expert"] = se_result
    weight_stats_path = output_dir / "phase3_single_expert_weight_stats.csv"
    se_result["weight_stats"].to_csv(weight_stats_path, index=False)
    logger.info(
        "Wrote weight-stats diagnostic table: %s (mean of per-chunk mean weight=%.4f)",
        weight_stats_path, se_result["weight_stats"]["weight_mean"].mean(),
    )

    _rewrite_combined_tables(results, combined_path, summary_path, args.seed)

    se_duration = time.time() - se_start
    logger.info("=== Phase 3 single-expert soft-weighted pipeline complete in %.2fs ===", se_duration)

    if args.skip_mixture:
        return

    mix_log_path = setup_logging("phase4_mixture")
    mix_start = time.time()
    logger.info("=== Phase 4 mixture-of-experts pipeline starting ===")
    logger.info("Log file: %s", mix_log_path)

    mix_result = run_mixture(
        stream.X, stream.y,
        initial_train_frac=args.initial_train_frac,
        chunk_size=args.single_expert_chunk_size,
        hazard_rate=args.single_expert_hazard_rate,
        boost_rounds_per_chunk=args.single_expert_boost_rounds_per_chunk,
        K=args.mixture_K,
        spawn_threshold=args.mixture_spawn_threshold,
        probation_window=args.mixture_probation_window,
        seed=args.seed,
        window_size=args.window_size,
    )
    results["mixture"] = mix_result

    spawn_log_path = output_dir / "phase4_mixture_spawn_log.csv"
    mix_result["spawn_log"].to_csv(spawn_log_path, index=False)
    logger.info(
        "Wrote spawn log: %s (%d spawns: %d promoted, %d discarded, spurious_spawn_rate=%.3f)",
        spawn_log_path, mix_result["metadata"]["n_spawns_total"],
        mix_result["metadata"]["n_promoted"], mix_result["metadata"]["n_discarded"],
        mix_result["metadata"]["spurious_spawn_rate"],
    )

    _rewrite_combined_tables(results, combined_path, summary_path, args.seed)

    mix_duration = time.time() - mix_start
    logger.info("=== Phase 4 mixture-of-experts pipeline complete in %.2fs ===", mix_duration)


if __name__ == "__main__":
    main()
