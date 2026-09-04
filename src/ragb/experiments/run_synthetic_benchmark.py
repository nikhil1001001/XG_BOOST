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

import pandas as pd

from ragb.baselines.periodic_retrain import run_periodic_retrain
from ragb.baselines.static_xgb import run_static_xgb
from ragb.data.synthetic_generator import SyntheticRegimeGenerator, SyntheticStreamConfig
from ragb.utils.logging_config import get_logger, setup_logging

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1: synthetic benchmark, static_xgb + periodic_retrain baselines")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=20_000)
    p.add_argument("--n-features", type=int, default=10)
    p.add_argument("--n-eras", type=int, default=3)
    p.add_argument("--hazard-rate", type=float, default=1.0 / 1500.0)
    p.add_argument("--min-run-length", type=int, default=800)
    p.add_argument("--initial-train-frac", type=float, default=0.2)
    p.add_argument("--retrain-every", type=int, default=1000)
    p.add_argument("--window-size", type=int, default=500)
    p.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "results" / "tables"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_path = setup_logging("phase1_synthetic_baselines")
    logger = get_logger(__name__)

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_rows = []
    for method, res in results.items():
        df = res["pr_auc_over_time"].copy()
        df["method"] = method
        combined_rows.append(df)
    combined = pd.concat(combined_rows, ignore_index=True)
    combined_path = output_dir / "phase1_baselines_pr_auc_over_time.csv"
    combined.to_csv(combined_path, index=False)
    logger.info("Wrote windowed PR-AUC table: %s (%d rows)", combined_path, len(combined))

    summary_rows = []
    for method, res in results.items():
        meta = res["metadata"]
        summary_rows.append({
            "method": method,
            "mean_pr_auc": res["pr_auc_over_time"]["pr_auc"].dropna().mean(),
            "n_retrains": meta["n_retrains"],
            "total_boosting_rounds": meta["total_boosting_rounds"],
            "train_time_sec": meta["train_time_sec"],
            "seed": args.seed,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "phase1_baselines_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Wrote summary table: %s\n%s", summary_path, summary_df.to_string(index=False))

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


if __name__ == "__main__":
    main()
