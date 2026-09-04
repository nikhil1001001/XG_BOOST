"""Mixture-of-boosted-experts controller: spawn/promote/discard/prune at capacity K, plus posterior-weighted inference blending (Phase 4).

Design (an implementation-level fill-in for what Section 3/5 name conceptually but don't pin down as
an algorithm -- documented here and in the Phase 4 report):

- One expert (the "trunk") is always being continuously warm-started, exactly like Phase 3's
  `single_expert` (soft-weighted via BOCPD's run-length-posterior survival function) -- it represents
  "the model for whichever regime is currently believed active."
- A single shared, continuously-running BOCPD detector tracks the residual loss of the mixture's own
  blended predictions. When its post-break weight crosses `spawn_threshold`, a new CANDIDATE expert is
  spawned: a fresh booster trained from scratch on data from the spawn point onward (no soft-weighting
  needed -- by construction all of a candidate's training data postdates its own birth). At most one
  candidate is on probation at a time; a spawn trigger while one is already on probation is logged and
  ignored rather than starting a second, overlapping candidate.
- Both the trunk and the candidate (if any) train every chunk; the trunk never pauses just because a
  candidate exists (this sidesteps having to "catch the trunk up" if the candidate is later discarded).
- PROMOTION/DISCARD (Section 5's "rolling validation window"): after `probation_window` timesteps,
  the candidate is promoted (folded permanently into the active mixture, and the trunk pointer moves to
  it) if the blended prediction *including* the candidate had lower mean log-loss over the probation
  window than the counterfactual blend *without* it -- i.e. it demonstrably helped, evaluated against
  real labels, not just BOCPD's own internal posterior. Otherwise it's discarded and logged as a
  spurious spawn (Section 5's required spurious-spawn-rate metric).
- BLENDING (Section 3 point 3): every expert (frozen historical ones, the trunk, and a probationary
  candidate) has a fixed birth_time. BOCPD's run-length posterior at time t assigns probability mass to
  "the current regime started at t-r" for each run length r; partitioning r by which expert's
  [birth_time, next expert's birth_time) bracket t-r falls into, and summing posterior mass per
  partition, gives each expert a blend weight -- these sum to 1 automatically, and a fresh candidate's
  bin naturally gets more weight as the run-length posterior increasingly favors "recently reset" as
  more post-spawn data arrives, which is exactly the "soft blending during transition" behavior Section
  9 says prevents cold-start from tanking accuracy.
- PRUNING at capacity K: on promotion, if the number of *active* (non-probation) experts exceeds K,
  the oldest active expert (by birth_time) is archived (removed from the blend/training set). The
  trunk itself is never pruned.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

from ragb.bocpd.detector import BOCPD
from ragb.bocpd.signals import binary_log_loss, rolling_mean
from ragb.boosting.custom_objective import make_soft_weighted_logloss_obj
from ragb.boosting.single_expert import DEFAULT_XGB_PARAMS, survival_weights
from ragb.eval.metrics import windowed_pr_auc
from ragb.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MixtureExpert:
    id: int
    birth_time: int
    booster: xgb.Booster
    status: str  # "probation" | "active" | "archived"


def _blend_weights(experts: list[MixtureExpert], now: int, run_length_posterior: np.ndarray) -> dict[int, float]:
    """Partitions run-length posterior mass among experts by birth-time bins (see module docstring).
    Experts must be passed sorted by ascending birth_time; the newest gets run lengths [0, now-birth],
    each older expert gets the next slice back, and the oldest catches all remaining (unbounded) mass.
    """
    if len(experts) == 1:
        return {experts[0].id: 1.0}

    L = len(run_length_posterior)
    survival = np.cumsum(run_length_posterior[::-1])[::-1]  # survival[k] = P(r >= k)

    def surv_at(k: int) -> float:
        if k <= 0:
            return 1.0
        if k >= L:
            return 0.0
        return float(survival[k])

    sorted_experts = sorted(experts, key=lambda e: e.birth_time)
    weights = {}
    for i, expert in enumerate(sorted_experts):
        r_upper = now - expert.birth_time  # this expert's bin: r in (r_lower_exclusive, r_upper]
        if i + 1 < len(sorted_experts):
            r_lower = now - sorted_experts[i + 1].birth_time  # next (younger) expert's birth
        else:
            r_lower = -1  # oldest expert: catch-all down to r=0 as well if it's alone... handled below
        # mass in (r_lower, r_upper] = P(r >= r_lower+1) - P(r >= r_upper+1)
        mass = surv_at(r_lower + 1) - surv_at(r_upper + 1)
        weights[expert.id] = max(mass, 0.0)

    # Oldest expert also absorbs all mass beyond the posterior's tail (r >= its own upper bound),
    # already included via surv_at(r_upper+1) evaluating toward 0 for the oldest bin's upper edge
    # being effectively unbounded -- but the oldest expert's r_upper is finite (now - its birth_time),
    # so explicitly add whatever mass lies beyond that (older than even the oldest expert believes
    # itself to be, which can happen if the detector's posterior still has support beyond the
    # oldest expert's own birth, e.g. right after that expert's own promotion).
    oldest = sorted_experts[0]
    weights[oldest.id] += surv_at(now - oldest.birth_time + 1)

    total = sum(weights.values())
    if total <= 0:
        # Degenerate (shouldn't happen once posterior is any real distribution): fall back to the trunk.
        return {sorted_experts[-1].id: 1.0}
    return {k: v / total for k, v in weights.items()}


def decide_promotion(loss_with_candidate: list[float], loss_without_candidate: list[float]) -> tuple[bool, float, float]:
    """Promotion rule (Section 5's "rolling validation window"): promote iff the blend WITH the
    candidate had lower mean log-loss over the probation window than the counterfactual blend
    WITHOUT it. Pure function of the two loss sequences so it's directly testable against
    constructed fixtures, without running any real training (Section 13).
    """
    mean_with = float(np.mean(loss_with_candidate)) if loss_with_candidate else float("inf")
    mean_without = float(np.mean(loss_without_candidate)) if loss_without_candidate else float("inf")
    return mean_with < mean_without, mean_with, mean_without


def select_expert_to_prune(active_experts: list[MixtureExpert], K: int) -> MixtureExpert | None:
    """Recency-weighted pruning: if more than K experts are active, returns the oldest one (by
    birth_time) to archive; otherwise returns None. Pure function of expert metadata, directly
    testable against constructed fixtures (Section 13).
    """
    if len(active_experts) <= K:
        return None
    return min(active_experts, key=lambda e: e.birth_time)


def _blended_predict(experts: list[MixtureExpert], weights: dict[int, float], X) -> np.ndarray:
    dmat = xgb.DMatrix(X)
    total = np.zeros(len(X))
    for expert in experts:
        w = weights.get(expert.id, 0.0)
        if w > 0:
            total += w * expert.booster.predict(dmat)
    return total


def run_mixture(
    X: pd.DataFrame,
    y: pd.Series,
    initial_train_frac: float = 0.2,
    chunk_size: int = 500,
    hazard_rate: float = 0.0004,
    smoothing_window: int = 20,
    break_window: int = 15,
    spawn_threshold: float = 0.3,
    probation_window: int = 1000,
    K: int = 3,
    initial_boost_rounds: int = 100,
    boost_rounds_per_chunk: int = 20,
    xgb_params: dict | None = None,
    seed: int = 42,
    window_size: int = 500,
) -> dict:
    params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {}), "seed": seed}
    n = len(X)
    split = int(n * initial_train_frac)

    logger.info(
        "mixture: initial trunk training on first %d/%d rows, K=%d, hazard_rate=%.5f, spawn_threshold=%.2f, probation_window=%d",
        split, n, K, hazard_rate, spawn_threshold, probation_window,
    )
    t0 = time.time()
    dtrain_init = xgb.DMatrix(X.iloc[:split], label=y.iloc[:split])
    trunk_booster = xgb.train(params, dtrain_init, num_boost_round=initial_boost_rounds)
    total_train_time = time.time() - t0
    total_rounds = initial_boost_rounds

    next_id = 0
    trunk = MixtureExpert(id=next_id, birth_time=0, booster=trunk_booster, status="active")
    next_id += 1
    experts: list[MixtureExpert] = [trunk]
    candidate: MixtureExpert | None = None
    candidate_deadline: int | None = None
    candidate_window_loss_with: list[float] = []
    candidate_window_loss_without: list[float] = []

    detector = BOCPD(hazard_rate=hazard_rate)
    scores = np.full(n, np.nan)
    scores[:split] = trunk_booster.predict(dtrain_init)

    spawn_log = []
    n_suppressed_spawns = 0

    chunk_starts = list(range(split, n, chunk_size))
    logger.info("mixture: %d chunks, chunk_size=%d, boost_rounds_per_chunk=%d", len(chunk_starts), chunk_size, boost_rounds_per_chunk)

    for chunk_start in tqdm(chunk_starts, desc="mixture: chunks", mininterval=1.0):
        chunk_end = min(chunk_start + chunk_size, n)
        X_chunk = X.iloc[chunk_start:chunk_end]
        y_chunk = y.iloc[chunk_start:chunk_end]
        now = chunk_end - 1

        active_experts = [e for e in experts if e.status in ("active", "probation")]
        blend_weights = _blend_weights(active_experts, now=chunk_start - 1, run_length_posterior=detector.run_length_posterior)
        blended_scores = _blended_predict(active_experts, blend_weights, X_chunk)
        scores[chunk_start:chunk_end] = blended_scores

        # Track promotion/discard evidence for any candidate currently on probation, using the
        # counterfactual "what if we predicted without the candidate" blend on the SAME chunk/labels.
        if candidate is not None:
            experts_without_candidate = [e for e in active_experts if e.id != candidate.id]
            weights_without = _blend_weights(experts_without_candidate, now=chunk_start - 1, run_length_posterior=detector.run_length_posterior)
            scores_without = _blended_predict(experts_without_candidate, weights_without, X_chunk)
            candidate_window_loss_with.append(float(np.mean(binary_log_loss(y_chunk.to_numpy(), blended_scores))))
            candidate_window_loss_without.append(float(np.mean(binary_log_loss(y_chunk.to_numpy(), scores_without))))

        residual = binary_log_loss(y_chunk.to_numpy(), blended_scores)
        smoothed = rolling_mean(residual, smoothing_window)
        for x in smoothed:
            detector.update(float(x))

        # Spawn check (post-break weight on the shared detector), only if no candidate is already on probation.
        pbw = detector.post_break_weight(break_window)
        if pbw >= spawn_threshold and candidate is None:
            dcand = xgb.DMatrix(X_chunk, label=y_chunk)
            cand_booster = xgb.train(params, dcand, num_boost_round=initial_boost_rounds)
            candidate = MixtureExpert(id=next_id, birth_time=chunk_start, booster=cand_booster, status="probation")
            next_id += 1
            experts.append(candidate)
            candidate_deadline = chunk_start + probation_window
            candidate_window_loss_with, candidate_window_loss_without = [], []
            total_rounds += initial_boost_rounds
            logger.info("mixture: SPAWNED candidate id=%d at t=%d (post_break_weight=%.3f)", candidate.id, chunk_start, pbw)
        elif pbw >= spawn_threshold and candidate is not None:
            n_suppressed_spawns += 1
            logger.debug("mixture: spawn trigger at t=%d suppressed, candidate id=%d already on probation", chunk_start, candidate.id)

        # Train trunk (always) and candidate (if any) on this chunk.
        posterior = detector.run_length_posterior
        instance_times = np.arange(chunk_start, chunk_end)
        distances = now - instance_times
        trunk_weights = survival_weights(posterior, distances)
        t0 = time.time()
        dchunk = xgb.DMatrix(X_chunk, label=y_chunk)
        obj = make_soft_weighted_logloss_obj(trunk_weights)
        trunk.booster = xgb.train(params, dchunk, num_boost_round=boost_rounds_per_chunk, xgb_model=trunk.booster, obj=obj)
        total_train_time += time.time() - t0
        total_rounds += boost_rounds_per_chunk

        if candidate is not None:
            t0 = time.time()
            candidate.booster = xgb.train(params, dchunk, num_boost_round=boost_rounds_per_chunk, xgb_model=candidate.booster)
            total_train_time += time.time() - t0
            total_rounds += boost_rounds_per_chunk

        # Promotion/discard decision once the candidate's probation window has elapsed.
        if candidate is not None and chunk_end >= candidate_deadline:
            promoted, mean_loss_with, mean_loss_without = decide_promotion(candidate_window_loss_with, candidate_window_loss_without)
            spawn_log.append({
                "candidate_id": candidate.id,
                "birth_time": candidate.birth_time,
                "decided_at": chunk_end,
                "mean_loss_with_candidate": mean_loss_with,
                "mean_loss_without_candidate": mean_loss_without,
                "outcome": "promoted" if promoted else "discarded",
            })
            if promoted:
                candidate.status = "active"
                trunk = candidate
                logger.info(
                    "mixture: PROMOTED candidate id=%d (loss_with=%.4f < loss_without=%.4f)",
                    candidate.id, mean_loss_with, mean_loss_without,
                )
                active_now = [e for e in experts if e.status == "active"]
                to_prune = select_expert_to_prune(active_now, K)
                if to_prune is not None:
                    to_prune.status = "archived"
                    logger.info("mixture: PRUNED expert id=%d (birth_time=%d) at capacity K=%d", to_prune.id, to_prune.birth_time, K)
            else:
                experts.remove(candidate)
                logger.info(
                    "mixture: DISCARDED candidate id=%d (loss_with=%.4f >= loss_without=%.4f, spurious spawn)",
                    candidate.id, mean_loss_with, mean_loss_without,
                )
            candidate = None
            candidate_deadline = None
            candidate_window_loss_with, candidate_window_loss_without = [], []

    pr_auc_df = windowed_pr_auc(y.iloc[split:].to_numpy(), scores[split:], window_size)
    pr_auc_df["window_start"] += split
    pr_auc_df["window_end"] += split

    spawn_df = pd.DataFrame(spawn_log)
    n_promoted = int((spawn_df["outcome"] == "promoted").sum()) if len(spawn_df) else 0
    n_discarded = int((spawn_df["outcome"] == "discarded").sum()) if len(spawn_df) else 0
    n_spawns_total = n_promoted + n_discarded
    spurious_spawn_rate = n_discarded / n_spawns_total if n_spawns_total > 0 else float("nan")

    logger.info(
        "mixture: done, %d spawns (%d promoted, %d discarded, spurious_rate=%.3f), %d suppressed, "
        "%d active experts remain, total_rounds=%d, total_train_time=%.2fs, mean PR-AUC=%.4f",
        n_spawns_total, n_promoted, n_discarded, spurious_spawn_rate, n_suppressed_spawns,
        sum(1 for e in experts if e.status == "active"), total_rounds, total_train_time,
        pr_auc_df["pr_auc"].dropna().mean(),
    )

    return {
        "scores": scores,
        "pr_auc_over_time": pr_auc_df,
        "spawn_log": spawn_df,
        "metadata": {
            "method": "mixture",
            "train_rows": split,
            "eval_rows": n - split,
            "n_chunks": len(chunk_starts),
            "chunk_size": chunk_size,
            "hazard_rate": hazard_rate,
            "K": K,
            "spawn_threshold": spawn_threshold,
            "probation_window": probation_window,
            "n_spawns_total": n_spawns_total,
            "n_promoted": n_promoted,
            "n_discarded": n_discarded,
            "spurious_spawn_rate": spurious_spawn_rate,
            "n_suppressed_spawns": n_suppressed_spawns,
            "n_active_experts_final": sum(1 for e in experts if e.status == "active"),
            "n_retrains": len(chunk_starts) + 1 + n_spawns_total,
            "total_boosting_rounds": total_rounds,
            "train_time_sec": total_train_time,
            "params": params,
        },
    }
