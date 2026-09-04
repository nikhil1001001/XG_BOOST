"""Ablation sweeps: hard vs. soft cutover, mixture vs. single expert, K sweep (1/3/unbounded), signal choice.

Run on the synthetic benchmark (known ground truth, matching Section 6's "primary correctness check"
framing) rather than real data, since these ablations are about isolating one mechanism at a time --
exactly what a controlled synthetic generator is for.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ragb.baselines.static_xgb import run_static_xgb
from ragb.bocpd.detector import run_bocpd_on_signal
from ragb.bocpd.signals import binary_log_loss, feature_drift_kl_signal, rolling_mean
from ragb.boosting.mixture import run_mixture
from ragb.boosting.single_expert import run_single_expert
from ragb.data.synthetic_generator import SyntheticRegimeGenerator, SyntheticStreamConfig
from ragb.eval.metrics import detect_events_from_probability_trace, detection_lag_and_false_alarms
from ragb.utils.logging_config import get_logger, setup_logging

REPO_ROOT = Path(__file__).resolve().parents[3]

# K=999 stands in for "unbounded" -- no synthetic run here spawns anywhere near that many experts, so
# it behaves identically to a true cap-free mixture while reusing the same capacity-check code path.
UNBOUNDED_K = 999


def ablation_hard_vs_soft(stream, args, logger) -> pd.DataFrame:
    rows = []
    for hard in (False, True):
        label = "hard" if hard else "soft"
        logger.info("hard_vs_soft: running variant=%s", label)
        r = run_single_expert(
            stream.X, stream.y, initial_train_frac=args.initial_train_frac, chunk_size=args.chunk_size,
            hazard_rate=args.hazard_rate, boost_rounds_per_chunk=args.boost_rounds_per_chunk,
            seed=args.seed, window_size=args.window_size, hard_cutover=hard,
        )
        rows.append({
            "variant": label,
            "mean_pr_auc": r["pr_auc_over_time"]["pr_auc"].dropna().mean(),
            "total_boosting_rounds": r["metadata"]["total_boosting_rounds"],
            "mean_weight": r["weight_stats"]["weight_mean"].mean(),
        })
        logger.info("hard_vs_soft: variant=%s mean_pr_auc=%.4f mean_weight=%.4f", label, rows[-1]["mean_pr_auc"], rows[-1]["mean_weight"])
    return pd.DataFrame(rows)


def ablation_moe_vs_single(stream, args, logger) -> pd.DataFrame:
    logger.info("moe_vs_single: running single_expert")
    r_single = run_single_expert(
        stream.X, stream.y, initial_train_frac=args.initial_train_frac, chunk_size=args.chunk_size,
        hazard_rate=args.hazard_rate, boost_rounds_per_chunk=args.boost_rounds_per_chunk,
        seed=args.seed, window_size=args.window_size,
    )
    logger.info("moe_vs_single: running mixture (K=%d)", args.mixture_K)
    r_mix = run_mixture(
        stream.X, stream.y, initial_train_frac=args.initial_train_frac, chunk_size=args.chunk_size,
        hazard_rate=args.hazard_rate, boost_rounds_per_chunk=args.boost_rounds_per_chunk,
        K=args.mixture_K, spawn_threshold=args.mixture_spawn_threshold, probation_window=args.mixture_probation_window,
        seed=args.seed, window_size=args.window_size,
    )
    rows = [
        {
            "variant": "single_expert",
            "mean_pr_auc": r_single["pr_auc_over_time"]["pr_auc"].dropna().mean(),
            "total_boosting_rounds": r_single["metadata"]["total_boosting_rounds"],
            "spurious_spawn_rate": float("nan"),
        },
        {
            "variant": "mixture",
            "mean_pr_auc": r_mix["pr_auc_over_time"]["pr_auc"].dropna().mean(),
            "total_boosting_rounds": r_mix["metadata"]["total_boosting_rounds"],
            "spurious_spawn_rate": r_mix["metadata"]["spurious_spawn_rate"],
        },
    ]
    return pd.DataFrame(rows)


def ablation_k_sweep(stream, args, logger) -> pd.DataFrame:
    rows = []
    for K in (1, 3, UNBOUNDED_K):
        label = "unbounded" if K == UNBOUNDED_K else str(K)
        logger.info("k_sweep: running K=%s", label)
        r = run_mixture(
            stream.X, stream.y, initial_train_frac=args.initial_train_frac, chunk_size=args.chunk_size,
            hazard_rate=args.hazard_rate, boost_rounds_per_chunk=args.boost_rounds_per_chunk,
            K=K, spawn_threshold=args.mixture_spawn_threshold, probation_window=args.mixture_probation_window,
            seed=args.seed, window_size=args.window_size,
        )
        meta = r["metadata"]
        rows.append({
            "K": label,
            "mean_pr_auc": r["pr_auc_over_time"]["pr_auc"].dropna().mean(),
            "total_boosting_rounds": meta["total_boosting_rounds"],
            "n_spawns_total": meta["n_spawns_total"],
            "n_promoted": meta["n_promoted"],
            "spurious_spawn_rate": meta["spurious_spawn_rate"],
            "n_active_experts_final": meta["n_active_experts_final"],
        })
        logger.info("k_sweep: K=%s mean_pr_auc=%.4f n_active_final=%d", label, rows[-1]["mean_pr_auc"], rows[-1]["n_active_experts_final"])
    return pd.DataFrame(rows)


def ablation_signal_choice(stream, args, logger) -> pd.DataFrame:
    """Compares BOCPD detection quality (Phase 2's lag/false-alarm methodology) under the two
    candidate signals named in Section 3: residual-loss (per-instance, row granularity) vs.
    feature-drift KL (inherently a window/batch quantity -- see feature_drift_kl_signal's docstring
    -- so it's evaluated at chunk granularity, with breakpoints and max_lag converted into chunk
    units for a fair comparison at each signal's own natural resolution).
    """
    split = int(len(stream.X) * args.initial_train_frac)
    static_result = run_static_xgb(stream.X, stream.y, initial_train_frac=args.initial_train_frac, seed=args.seed, window_size=args.window_size)
    y_eval = stream.y.iloc[split:].to_numpy()
    true_breaks_eval = np.array([bp - split for bp in stream.breakpoints if bp > split])

    logger.info("signal_choice: evaluating residual_loss signal")
    residual = binary_log_loss(y_eval, static_result["scores"][split:])
    smoothed = rolling_mean(residual, args.smoothing_window)
    bocpd_res = run_bocpd_on_signal(smoothed, hazard_rate=args.hazard_rate, break_window=args.break_window)
    events_res = detect_events_from_probability_trace(bocpd_res["post_break_weight"], threshold=args.event_threshold)
    m_res = detection_lag_and_false_alarms(true_breaks_eval, events_res, max_lag=args.max_lag, n_timesteps=len(smoothed))
    m_res["signal"] = "residual_loss"
    m_res["granularity"] = "row"

    logger.info("signal_choice: evaluating feature_drift_kl signal (chunk_size=%d)", args.chunk_size)
    X_eval = stream.X.iloc[split:].reset_index(drop=True)
    kl_signal = feature_drift_kl_signal(X_eval, chunk_size=args.chunk_size)
    break_window_chunks = max(1, args.break_window // args.chunk_size) if args.break_window >= args.chunk_size else 1
    bocpd_kl = run_bocpd_on_signal(kl_signal, hazard_rate=args.hazard_rate, break_window=break_window_chunks)
    events_kl = detect_events_from_probability_trace(bocpd_kl["post_break_weight"], threshold=args.event_threshold)
    true_breaks_chunks = np.array(sorted(set(int(b) // args.chunk_size for b in true_breaks_eval)))
    max_lag_chunks = max(1, args.max_lag // args.chunk_size)
    m_kl = detection_lag_and_false_alarms(true_breaks_chunks, events_kl, max_lag=max_lag_chunks, n_timesteps=len(kl_signal))
    m_kl["signal"] = "feature_drift_kl"
    m_kl["granularity"] = "chunk"

    logger.info("signal_choice: residual_loss detected=%d/%d false_alarms=%d | feature_drift_kl detected=%d/%d false_alarms=%d",
                m_res["n_detected"], m_res["n_true_breaks"], m_res["n_false_alarms"],
                m_kl["n_detected"], m_kl["n_true_breaks"], m_kl["n_false_alarms"])

    return pd.DataFrame([m_res, m_kl])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5 ablations: hard/soft, MoE/single, K sweep, signal choice")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=20_000)
    p.add_argument("--n-features", type=int, default=10)
    p.add_argument("--n-eras", type=int, default=3)
    p.add_argument("--generator-hazard-rate", type=float, default=1.0 / 1500.0)
    p.add_argument("--min-run-length", type=int, default=800)
    p.add_argument("--initial-train-frac", type=float, default=0.2)
    p.add_argument("--window-size", type=int, default=500)
    p.add_argument("--hazard-rate", type=float, default=0.0004, help="BOCPD hazard rate used throughout the ablations")
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--boost-rounds-per-chunk", type=int, default=20)
    p.add_argument("--smoothing-window", type=int, default=20)
    p.add_argument("--break-window", type=int, default=15)
    p.add_argument("--event-threshold", type=float, default=0.3)
    p.add_argument("--max-lag", type=int, default=300)
    p.add_argument("--mixture-K", type=int, default=3)
    p.add_argument("--mixture-spawn-threshold", type=float, default=0.3)
    p.add_argument("--mixture-probation-window", type=int, default=1000)
    p.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "results" / "tables"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_path = setup_logging("phase5_ablations")
    logger = get_logger(__name__)

    run_start = time.time()
    logger.info("=== Phase 5 ablations starting ===")
    logger.info("Resolved config: %s", vars(args))
    logger.info("Log file: %s", log_path)

    stream_cfg = SyntheticStreamConfig(
        n_samples=args.n_samples, n_features=args.n_features, n_eras=args.n_eras,
        hazard_rate=args.generator_hazard_rate, min_run_length=args.min_run_length, seed=args.seed,
    )
    stream = SyntheticRegimeGenerator(stream_cfg).generate()
    logger.info("Data shape: X=%s, %d ground-truth breakpoints, fraud rate=%.4f", stream.X.shape, len(stream.breakpoints), stream.y.mean())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("--- Ablation 1: hard vs. soft cutover ---")
    df1 = ablation_hard_vs_soft(stream, args, logger)
    df1.to_csv(output_dir / "phase5_ablation_hard_vs_soft.csv", index=False)
    logger.info("\n%s", df1.to_string(index=False))

    logger.info("--- Ablation 2: mixture-of-experts vs. single continuously-warm-started expert ---")
    df2 = ablation_moe_vs_single(stream, args, logger)
    df2.to_csv(output_dir / "phase5_ablation_moe_vs_single.csv", index=False)
    logger.info("\n%s", df2.to_string(index=False))

    logger.info("--- Ablation 3: K sweep (1 / 3 / unbounded) ---")
    df3 = ablation_k_sweep(stream, args, logger)
    df3.to_csv(output_dir / "phase5_ablation_k_sweep.csv", index=False)
    logger.info("\n%s", df3.to_string(index=False))

    logger.info("--- Ablation 4: detection-signal choice (residual-loss vs. feature-drift KL) ---")
    df4 = ablation_signal_choice(stream, args, logger)
    df4.to_csv(output_dir / "phase5_ablation_signal_choice.csv", index=False)
    logger.info("\n%s", df4.to_string(index=False))

    duration = time.time() - run_start
    logger.info("=== Phase 5 ablations complete in %.2fs ===", duration)


if __name__ == "__main__":
    main()
