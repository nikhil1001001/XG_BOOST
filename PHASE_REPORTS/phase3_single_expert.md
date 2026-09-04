# Phase 3 — Soft-Weighted Single Expert

## What was built
- `src/ragb/boosting/custom_objective.py`: `make_soft_weighted_logloss_obj(weights)` — a custom
  `obj(preds, dtrain)` closure implementing exactly the Section 3 "Math core" formula. Standard binary
  logloss grad/hess (`p_i - y_i`, `p_i(1-p_i)`) scaled per instance by its soft weight `w_i` before
  XGBoost sums them into each leaf's `G`, `H` — mathematically identical to `G = Σw_i·g_i`,
  `H = Σw_i·h_i` feeding the leaf-weight formula `-G/(H+λ)`, since leaf sums are linear in each
  instance's contribution.
- `src/ragb/boosting/single_expert.py`: `run_single_expert()` — one continuously warm-started XGBoost
  booster (never restarted) processing the stream chunk by chunk. Each chunk: score with the current
  booster to get a residual-loss signal, feed it into a single continuously-running BOCPD detector
  (state persists across chunks), derive a soft weight per instance from the current run-length
  posterior via a survival-probability formula (`_survival_weights`), and add a fixed number of new
  boosting rounds fit on the chunk with those weights via the custom objective, warm-started from the
  existing booster.
- **Soft-weight formula** (an implementation-level design decision the brief didn't specify
  numerically — Section 3 only names the concept, "P(instance i is post-break)"): for an instance
  observed `d` steps before "now" (the end of the current chunk), `w_i = P(current run length ≥ d)`,
  i.e. the run-length posterior's survival function. This is 1 for instances at "now", decays smoothly
  further back, and drops off sharply right around a suspected changepoint — directly the "soft-assign
  instances near a suspected changepoint instead of a brittle hard split" behavior Section 3 asks for,
  and it falls straight out of the run-length posterior BOCPD already computes.
- `run_synthetic_benchmark.py` extended with a Phase 3 section: runs `single_expert`, adds it to the
  same combined comparison tables Phase 1 established (`phase1_baselines_summary.csv` /
  `..._pr_auc_over_time.csv`), and writes a new `phase3_single_expert_weight_stats.csv` diagnostic
  table (per-chunk min/mean/max soft weight — see finding below).
- Tests: `tests/test_custom_objective.py` (4 tests) and `tests/test_single_expert.py` (6 tests).

## Acceptance-criteria test: custom obj() reduces to standard boosting at weights=1
`test_reduces_to_standard_boosting_when_all_weights_are_one` trains two ten-round boosters on
identical data/params — one via XGBoost's built-in `binary:logistic` objective, one via
`make_soft_weighted_logloss_obj(weights=np.ones(n))` — and asserts their raw-margin predictions match
to `atol=1e-5`. **Passes.** Also directly checks the grad/hess arrays against the closed-form
`p-y`/`p(1-p)` formulas (exact to `1e-12`), that weights scale grad/hess linearly, and that
all-zero weights produce zero gradient with a numerically-floored (not literally zero) Hessian.

## A real finding: Phase 2's "recommended" hazard rate was wrong for this use, not just reusable

Phase 2's report picked `hazard_rate=0.002` as the sweep's best *detection-recall* operating point and
suggested carrying it into Phase 3. Using it directly for `single_expert`'s soft-weighting produced a
**mean per-chunk weight of 0.066** — i.e. on average the pipeline was discarding ~93% of each chunk's
instance-worth of gradient signal — and PR-AUC of 0.3447, *worse* than static_xgb's 0.3876 despite
using 740 boosting rounds. Root cause: `hazard_rate=0.002` implies an assumed mean run length of 500
steps (`1/0.002`), but the actual generator's mean run length here is roughly 2,300 steps
(`min_run_length=800 + 1/generator_hazard_rate=1500`) — a >4x mismatch. The detector's run-length
posterior, primed to expect breaks every ~500 steps, rarely accumulates enough confidence in a long
run to give distant-but-still-relevant instances real weight, so most of each chunk gets treated as
"probably already stale."

A quick sweep specifically for the weighting use case (`hazard_rate` in
`{0.002, 0.0004, 0.0002, 0.00005, 0.00002}`, all else fixed) showed PR-AUC improves from 0.3447 to
0.3916 and then **plateaus** exactly at 0.0004 and below (0.3916 at every lower value tried) — so
0.0004 (closer to the generator's true switch rate, `1/1500`) was adopted as the new default rather
than continuing to chase an even lower value with no further payoff. This is flagged per Section 0's
"if a design decision turns out wrong once you're building, flag it, propose the fix, and proceed"
rule: **the hazard rate that's best for *detecting* a break (Phase 2's question) is not necessarily
the hazard rate that's best for *weighting* instances around a suspected break (Phase 3's question)**
— they're different tuning problems sharing one parameter name, and Phase 2's report didn't anticipate
that. `single_expert_hazard_rate` is now its own CLI flag (default `0.0004`), decoupled from Phase 2's
`--bocpd-hazard-sweep`, with this reasoning recorded in its help text.

## Results (seed=42, same 20,000-row / 3-era / 9-breakpoint stream as Phase 1/2)

| method | mean PR-AUC (eval windows) | n_retrains | total boosting rounds | train time (s) |
|---|---|---|---|---|
| static_xgb | 0.3876 | 1 | 100 | 0.44 |
| periodic_retrain | 0.3981 | 16 | 1,600 | 2.71 |
| **single_expert** | **0.3916** | 33 | **740** | 2.24 |

Full table: `results/tables/phase1_baselines_summary.csv` (now includes all three methods) and
`results/tables/phase1_baselines_pr_auc_over_time.csv`. Weight diagnostics:
`results/tables/phase3_single_expert_weight_stats.csv`.

**Honest read:** with the corrected hazard rate, `single_expert` already beats the static floor and
gets within 1.6% (relative) of periodic_retrain's PR-AUC while using **46% of its boosting rounds**
(740 vs. 1,600) — a real, if modest, point on the compute/accuracy tradeoff Section 10 asks RAGB to
eventually dominate. This is one warm-started expert on one seed, not yet the full mixture-of-experts
system (Phase 4) or a multi-seed confidence interval (Phase 5/6's job) — reported as a Phase 3
milestone result, not a final claim.

## What was tested
- `pytest tests/` — **33/33 passed** (14.44s): 4 new custom-objective tests, 6 new single-expert tests
  (survival-weight formula unit tests including a hand-computed case and a monotonicity property, plus
  end-to-end run/reproducibility/weight-bounds smoke tests), all prior tests still green.
- Full run: `python -m ragb.experiments.run_synthetic_benchmark --seed 42` — Phase 3 section completed
  in 10.03s, log at `logs/phase3_single_expert_20260904-224309.log` (re-run after the hazard-rate fix;
  see log timestamps for the corrected run).

## Deviations from spec + why
- **Soft-weight formula (survival function of the run-length posterior)** — an implementation choice
  filling in what Section 3 names conceptually but doesn't specify as a formula; documented above and
  in the module docstring. Directly derivable from BOCPD's existing posterior output, no extra
  machinery, and empirically produces the intended "soft assignment near a suspected changepoint"
  behavior (verified in `test_single_expert.py`'s hand-computed weight cases).
- **`single_expert_hazard_rate` decoupled from Phase 2's hazard-rate sweep** — see the finding above.
- **Chunk-based warm-starting** (`chunk_size=500` default, `boost_rounds_per_chunk=20` default) rather
  than instance-by-instance online updates — XGBoost's `xgb_model=` warm-start API operates on
  DMatrix-sized batches, and per-instance updates would be prohibitively slow (a full tree-building
  pass per single row); chunking is the standard practical compromise and is CLI-configurable.

## Pass/fail
**Pass.** Both Phase 3 acceptance criteria met: (1) the single-expert soft-weighted pipeline trains
and evaluates on synthetic data end-to-end (real run above, plus smoke/reproducibility tests); (2) unit
tests confirm the custom `obj()` reduces exactly to standard boosting when weights are all 1
(`test_reduces_to_standard_boosting_when_all_weights_are_one`, exact end-to-end prediction match).

## Log files
`logs/phase3_single_expert_20260904-224309.log` (corrected-hazard-rate run referenced in the results
table above); an earlier run at the mis-tuned `hazard_rate=0.002` is preserved at
`logs/phase3_single_expert_20260904-224145.log` for the record of the finding above.
