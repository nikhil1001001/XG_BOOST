# Phase 1 — Foundations

## What was built
- `src/ragb/utils/logging_config.py`: `setup_logging(phase)` (stdout + timestamped file handler under
  `logs/`) and `get_logger(name)`, per Section 12.
- `src/ragb/data/synthetic_generator.py`: `SyntheticRegimeGenerator` — simulates a fraud stream as a
  sequence of "eras," each with its own sparse logistic generative rule over a shared feature
  distribution (concept drift, matching the "fraud tactics shift" framing in Section 1, and
  deliberately *not* covariate/feature-distribution drift, which is a separate phenomenon). Run
  lengths are drawn from a constant-hazard (geometric) schedule — the same hazard assumption BOCPD
  itself uses in Phase 2, so Phase 2's validation is a fair test of that assumption rather than a
  mismatched one. Ground truth breakpoints are stored as exact indices where the active era changes.
  Also added `generate_single_changepoint()`, a convenience two-era/one-breakpoint generator for
  Phase 2's BOCPD sanity check (Section 13's `test_bocpd.py` needs a single known changepoint case).
- `src/ragb/eval/metrics.py`: `pr_auc()` and `windowed_pr_auc()` (non-overlapping windows over the
  stream). Only what Phase 1 needs; detection lag / false-alarm rate / TS-AUC are deferred to the
  phases that actually consume them (Phase 2 and Phase 5/6 respectively), per the phase-by-phase rule
  in Section 0 rather than building ahead of need.
- `src/ragb/baselines/static_xgb.py`: trains once on the first `initial_train_frac` of the stream,
  scores everything after with no further updates (the "stale model" floor from Section 1).
- `src/ragb/baselines/periodic_retrain.py`: walk-forward, cumulative full retrain every
  `retrain_every` rows (tracks `n_retrains` / `total_boosting_rounds` for the Section 8 compute-cost
  comparison against RAGB later).
- `src/ragb/experiments/run_synthetic_benchmark.py`: CLI entry point (with `--seed` per Section 0b's
  recommendation) wiring generator + both baselines together, writing `results/tables/` CSVs/JSON and
  a `logs/` file. This script is meant to be extended in later phases (BOCPD-aware baselines, RAGB
  itself) rather than replaced, so all synthetic-benchmark results stay in one comparable table.
- Tests: `tests/test_synthetic_generator.py` (6 tests — seed reproducibility, differing seeds differ,
  breakpoints exactly match actual regime-id changes, all configured eras get used on a long-enough
  stream, output shapes/label domain, single-changepoint convenience generator) and
  `tests/test_baselines_smoke.py` (3 tests — both baselines run end-to-end without error and produce
  non-degenerate PR-AUC, and `static_xgb` is deterministic given a fixed seed).

## What was tested
- `pytest tests/` — **9/9 passed** (5.92s).
- Full run: `python -m ragb.experiments.run_synthetic_benchmark --seed 42` (n=20,000, 10 features, 3
  eras, hazard_rate=1/1500) — completed in 3.69s, log at
  `logs/phase1_synthetic_baselines_20260904-222610.log`.

## Results (seed=42, n=20,000, 3 eras, 9 ground-truth breakpoints, fraud rate 4.91%)

| method | mean PR-AUC (eval windows) | n_retrains | total boosting rounds | train time (s) |
|---|---|---|---|---|
| static_xgb | 0.3876 | 1 | 100 | 0.43 |
| periodic_retrain | 0.3981 | 16 | 1,600 | 2.80 |

Full windowed breakdown in `results/tables/phase1_baselines_pr_auc_over_time.csv`; ground-truth
breakpoints in `results/tables/phase1_ground_truth_breakpoints.json`.

periodic_retrain modestly outperforms static_xgb (+0.0105 mean PR-AUC) at 16x the training compute —
directionally exactly what's expected (a stale model degrades across regime changes it never adapts
to; retraining recovers some of that, at a real compute cost). This 16x compute for a ~2.7% relative
PR-AUC gain is the concrete baseline RAGB needs to beat on the Pareto frontier in Phase 5/6 (Section
10) — better-or-equal accuracy at meaningfully less amortized compute than periodic_retrain.

## Deviations from spec + why
- **`initial_train_frac=0.2` and `retrain_every=1000` as defaults** — not specified numerically in the
  brief. Chose 20% initial training window (large enough for a stable first model, small enough to
  leave most of the stream for evaluation) and a 1000-row retrain cadence (yields double-digit retrain
  cycles on a 20k-row stream, enough to see periodic_retrain's compute-cost tradeoff without being
  vacuously fine-grained). Both are CLI-overridable flags on `run_synthetic_benchmark.py`.
- **`windowed_pr_auc` uses non-overlapping fixed windows**, not a rolling one — simpler and sufficient
  for Phase 1's acceptance bar ("produce PR-AUC over time without error"); can revisit if Phase 5/6's
  Pareto-frontier reporting wants finer temporal resolution.
- No deviation on the generator's core requirements: ≥3 distinct eras (configurable, default 3),
  seeded reproducibility (unit-tested), recoverable ground truth (unit-tested exact match against
  actual regime-id transitions), configurable hazard/switch schedule (`hazard_rate`,
  `min_run_length` CLI/config params).

## Pass/fail
**Pass.** Both Phase 1 acceptance criteria met: (1) synthetic generator produces reproducible, seeded
data with recoverable ground truth — verified in `test_synthetic_generator.py`; (2) both baselines
train and produce PR-AUC over time without error — verified in `test_baselines_smoke.py` and the real
end-to-end run above.

## Log file
`logs/phase1_synthetic_baselines_20260904-222610.log`
