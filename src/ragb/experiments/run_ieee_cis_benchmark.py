"""Entry point: run the full RAGB pipeline + all baselines on real data resolved via the Section 6a cascade.

Named `run_ieee_cis_benchmark.py` per Section 4's repo layout, but actually runs on WHICHEVER dataset
`real_data_loader.py` resolves -- Phase 5a lets the cascade start from the top (lands on ULB Credit
Card Fraud, no auth needed); Phase 5b passes `--preferred ieee_cis` to target the primary dataset
directly, falling through to Elliptic if Kaggle credentials are unavailable. Which source actually
ran is always in the metadata, the log, and the results table -- never silently swapped in.

Real data has no ground-truth breakpoints (unlike the synthetic benchmark), so the BOCPD section here
reports descriptive event counts/positions only, not detection lag or false-alarm rate against a
known truth.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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
from ragb.data.real_data_loader import RealDataUnavailableError, load_real_dataset
from ragb.eval.metrics import detect_events_from_probability_trace
from ragb.utils.logging_config import get_logger, setup_logging

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-data benchmark (Section 6a cascade): all baselines + RAGB pipeline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--phase-label", type=str, default="phase5a_real_data", help="log-file/table-file prefix, e.g. phase5a_real_data or phase5b_real_data")
    p.add_argument("--preferred", type=str, default=None, help="Section 6a cascade start point, e.g. 'ieee_cis' for Phase 5b")
    p.add_argument("--initial-train-frac", type=float, default=0.1)
    p.add_argument("--retrain-every", type=int, default=10_000)
    p.add_argument("--window-size", type=int, default=5_000)
    p.add_argument("--hazard-rate", type=float, default=0.0004)
    p.add_argument("--smoothing-window", type=int, default=50)
    p.add_argument("--break-window", type=int, default=15)
    p.add_argument("--event-threshold", type=float, default=0.3)
    p.add_argument("--single-expert-chunk-size", type=int, default=5_000)
    p.add_argument("--single-expert-boost-rounds-per-chunk", type=int, default=20)
    p.add_argument("--skip-mixture", action="store_true")
    p.add_argument("--mixture-K", type=int, default=3)
    p.add_argument("--mixture-spawn-threshold", type=float, default=0.3)
    p.add_argument("--mixture-probation-window", type=int, default=20_000)
    p.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "results" / "tables"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_path = setup_logging(args.phase_label)
    logger = get_logger(__name__)

    run_start = time.time()
    logger.info("=== Real-data benchmark run starting (label=%s) ===", args.phase_label)
    logger.info("Resolved config: %s", vars(args))
    logger.info("Log file: %s", log_path)

    try:
        X, y, data_meta = load_real_dataset(preferred=args.preferred)
    except RealDataUnavailableError as e:
        logger.error("All real-data sources failed: %s", e)
        raise

    logger.info("Resolved real-data source: %s (%s)", data_meta["source"], data_meta)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep only numeric-safe columns for XGBoost training (category dtype is fine; drop identifier-
    # like columns real loaders may have left in that aren't informative, e.g. none currently, but
    # this guards against future loader additions leaking a raw id column into training).
    n = len(X)
    split = int(n * args.initial_train_frac)
    logger.info("Data shape: X=%s y=%s, fraud/positive rate=%.5f, initial_train_frac=%.2f -> %d train rows", X.shape, y.shape, y.mean(), args.initial_train_frac, split)

    results = {}
    results["static_xgb"] = run_static_xgb(X, y, initial_train_frac=args.initial_train_frac, seed=args.seed, window_size=args.window_size)
    results["periodic_retrain"] = run_periodic_retrain(X, y, initial_train_frac=args.initial_train_frac, retrain_every=args.retrain_every, seed=args.seed, window_size=args.window_size)
    results["sliding_window_retrain"] = run_sliding_window_retrain(X, y, initial_train_frac=args.initial_train_frac, retrain_every=args.retrain_every, seed=args.seed, window_size=args.window_size)
    results["adwin_online"] = run_adwin_online(X, y, initial_train_frac=args.initial_train_frac, seed=args.seed, window_size=args.window_size)
    results["single_expert"] = run_single_expert(
        X, y, initial_train_frac=args.initial_train_frac, chunk_size=args.single_expert_chunk_size,
        hazard_rate=args.hazard_rate, boost_rounds_per_chunk=args.single_expert_boost_rounds_per_chunk,
        seed=args.seed, window_size=args.window_size,
    )
    if not args.skip_mixture:
        results["mixture"] = run_mixture(
            X, y, initial_train_frac=args.initial_train_frac, chunk_size=args.single_expert_chunk_size,
            hazard_rate=args.hazard_rate, boost_rounds_per_chunk=args.single_expert_boost_rounds_per_chunk,
            K=args.mixture_K, spawn_threshold=args.mixture_spawn_threshold, probation_window=args.mixture_probation_window,
            seed=args.seed, window_size=args.window_size,
        )

    combined_rows = []
    for method, res in results.items():
        df = res["pr_auc_over_time"].copy()
        df["method"] = method
        combined_rows.append(df)
    combined = pd.concat(combined_rows, ignore_index=True)
    combined_path = output_dir / f"{args.phase_label}_pr_auc_over_time.csv"
    combined.to_csv(combined_path, index=False)

    summary_rows = []
    for method, res in results.items():
        meta = res["metadata"]
        summary_rows.append({
            "method": method,
            "mean_pr_auc": res["pr_auc_over_time"]["pr_auc"].dropna().mean(),
            "n_retrains": meta["n_retrains"],
            "total_boosting_rounds": meta["total_boosting_rounds"],
            "train_time_sec": meta["train_time_sec"],
            "data_source": data_meta["source"],
            "seed": args.seed,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / f"{args.phase_label}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Wrote summary table: %s\n%s", summary_path, summary_df.to_string(index=False))

    if "mixture" in results:
        spawn_log_path = output_dir / f"{args.phase_label}_mixture_spawn_log.csv"
        results["mixture"]["spawn_log"].to_csv(spawn_log_path, index=False)
        logger.info("Wrote mixture spawn log: %s", spawn_log_path)

    # BOCPD descriptive pass (no ground truth on real data -- event count/positions only).
    logger.info("Running BOCPD descriptive pass on the static_xgb residual-loss signal (no ground truth available on real data)...")
    y_eval = y.iloc[split:].to_numpy()
    residual = binary_log_loss(y_eval, results["static_xgb"]["scores"][split:])
    smoothed = rolling_mean(residual, args.smoothing_window)
    bocpd_result = run_bocpd_on_signal(smoothed, hazard_rate=args.hazard_rate, break_window=args.break_window)
    events = detect_events_from_probability_trace(bocpd_result["post_break_weight"], threshold=args.event_threshold)
    bocpd_summary = {
        "hazard_rate": args.hazard_rate,
        "smoothing_window": args.smoothing_window,
        "break_window": args.break_window,
        "event_threshold": args.event_threshold,
        "n_events_detected": len(events),
        "event_positions_row_index": (events + split).tolist(),
        "n_eval_rows": len(smoothed),
    }
    bocpd_path = output_dir / f"{args.phase_label}_bocpd_events.json"
    with open(bocpd_path, "w") as f:
        json.dump(bocpd_summary, f, indent=2)
    logger.info("BOCPD descriptive pass: %d events detected over %d eval rows. Wrote: %s", len(events), len(smoothed), bocpd_path)

    data_meta_path = output_dir / f"{args.phase_label}_data_source.json"
    with open(data_meta_path, "w") as f:
        json.dump(data_meta, f, indent=2, default=str)
    logger.info("Wrote data-source metadata: %s", data_meta_path)

    duration = time.time() - run_start
    logger.info("=== Real-data benchmark run (label=%s) complete in %.2fs ===", args.phase_label, duration)


if __name__ == "__main__":
    main()
