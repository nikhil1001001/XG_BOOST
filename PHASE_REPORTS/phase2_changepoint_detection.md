# Phase 2 — Changepoint Detection

## What was built
- `src/ragb/bocpd/detector.py`: `BOCPD` class — standard Adams & MacKay (2007) run-length posterior
  tracker with a constant hazard rate and a Normal-Inverse-Gamma conjugate predictive model (Student-t
  predictive distribution) for a scalar real-valued signal. Includes probability-mass pruning so
  per-step cost stays roughly bounded on long streams rather than growing with total elapsed time.
  Exposes `map_run_length()`, `changepoint_probability()`, and `post_break_weight(window)` (see
  deviation note below on why the latter is the one that matters). `run_bocpd_on_signal()` is a bulk
  convenience wrapper for running the detector over a full signal and getting back per-timestep
  trajectories.
- `src/ragb/bocpd/signals.py`: `binary_log_loss()` (raw per-instance residual-loss signal, Section 3
  point 1) and `rolling_mean()` (smooths it into something closer to what the Normal predictive model
  assumes). `feature_drift_kl_signal()` is a documented `NotImplementedError` stub — deferred to Phase
  5's signal-choice ablation, which is the first thing that actually needs it.
- `src/ragb/eval/metrics.py` additions: `detect_events_from_probability_trace()` (collapses a
  probability trace into discrete detection events) and `detection_lag_and_false_alarms()` (matches
  detections to true breakpoints within a max-lag window, reports lag/miss/false-alarm counts).
- `run_synthetic_benchmark.py` extended with a `run_bocpd_validation()` function and a Phase 2 section
  in `main()`: builds the residual-loss signal from the Phase-1 static_xgb classifier's out-of-sample
  predictions, smooths it, sweeps BOCPD's hazard rate, and reports detection lag / false-alarm rate
  per hazard rate plus a lag-vs-false-alarm-rate figure. `--skip-bocpd` flag added for a Phase-1-only
  run.
- Tests: `tests/test_bocpd.py` (7 tests) and `tests/test_metrics.py` (7 tests, covering the new
  detection-metrics functions plus the Phase-1 PR-AUC functions that didn't have a dedicated test file
  yet).

## A real finding, not a bug: `changepoint_probability()` (raw P(r_t=0)) is mathematically pinned to the hazard rate

While writing the first version of `test_bocpd.py`, a test asserting that `changepoint_probability()`
spikes near a known changepoint **failed** — its value was a suspiciously constant `0.0040`
(`= 1/250`, the hazard rate used) at every single timestep, on both sides of the break. This turned
out not to be an implementation bug but a real, provable property of constant-hazard BOCPD: because
`P(r_t=0, x_1:t) = hazard_rate * S` and the full normalizer is *also* exactly `S` (both derive from
the same marginal evidence term), `P(r_t=0 | x_1:t)` collapses to the hazard rate constant
identically, for any data, at every timestep — confirmed both by hand-deriving the algebra and
empirically (`np.testing.assert_allclose(changepoint_prob, hazard_rate, atol=1e-9)` passes exactly).

The rest of the run-length posterior (r > 0) is **not** similarly degenerate and is exactly where the
real signal lives — confirmed the MAP run length correctly collapses to a low value within ~10 steps
of a true break, and `post_break_weight(window)` (aggregate posterior mass on "recently reset"
hypotheses) spikes sharply and genuinely (0.019 baseline -> 0.999 right at a known break in the
Gaussian-mean-shift test fixture). Fixed by: documenting the identity explicitly in
`changepoint_probability()`'s docstring, and switching all detection logic (in both the detector's own
tests and the Phase 2 experiment) to use `post_break_weight()` instead. This is worth flagging per
Section 0's "if a design decision turns out wrong once you're building, flag it and proceed" rule —
the brief's Section 3 doesn't specify which posterior quantity to threshold on, and the naive choice
(raw `P(r_t=0)`) turns out to be a dead end.

## What was tested
- `pytest tests/` — **23/23 passed** (7.21s), including the new 7 BOCPD tests (posterior sums to 1
  at every step, MAP run length resets near a known changepoint, the hazard-rate-identity property
  above, `post_break_weight` spike detection within tolerance, detection-lag/false-alarm computation
  on a known changepoint, post-break weight ordering, and low false-alarm count on a stationary
  signal) and 7 new metrics tests.
- Full run: `python -m ragb.experiments.run_synthetic_benchmark --seed 42` (same 20,000-row, 3-era,
  9-breakpoint stream as Phase 1) — Phase 2 section completed in 40.25s, logs at
  `logs/phase2_bocpd_hazard_sweep_20260904-223346.log`.

## Results — hazard-rate sweep (seed=42, static_xgb residual-loss signal, 20-step rolling mean, break_window=15, event_threshold=0.3, max_lag=300, 8 of 9 breakpoints evaluable after the classifier's training cutoff)

| hazard_rate | detected | missed | mean lag | false alarms | false-alarm rate /1000 |
|---|---|---|---|---|---|
| 0.0100 | 6/8 | 2 | 103.5 | 103 | 6.44 |
| 0.0040 | 7/8 | 1 | 126.0 | 108 | 6.75 |
| 0.0020 | 7/8 | 1 | 116.3 | 89 | 5.56 |
| 0.0010 | 6/8 | 2 | 96.3 | 76 | 4.75 |
| 0.0004 | 6/8 | 2 | 76.3 | 71 | 4.44 |
| 0.0002 | 6/8 | 2 | 66.8 | 80 | 5.00 |

Full table: `results/tables/phase2_bocpd_hazard_sweep.csv`. Figure:
`results/figures/phase2_hazard_sweep.png`.

**Honest read of these numbers:** detection works (6-7 of 8 breaks caught, well within the 300-step
max-lag window against an average inter-break spacing of ~2,200 steps), but the false-alarm rate is
genuinely high (~0.4-0.7% of timesteps) at every hazard rate tried, and both lag and false-alarm-rate
have a noisier, less monotonic relationship with `hazard_rate` than textbook BOCPD demos (which
typically use clean, low-noise real-valued signals) show. This traces to the residual-loss signal
itself: at the generator's ~5% base fraud rate, per-instance binary log-loss is dominated by Bernoulli
sampling noise (whether the 1-in-20 fraud case happens to land inside a given smoothing window swings
the rolling-mean signal a lot), which is exactly Section 9's expected "false alarms from a poorly
tuned hazard-rate prior" failure mode — reported here plainly rather than hidden or hand-tuned away.
**Recommended default carried into Phase 3:** `hazard_rate=0.002` (best detection recall in the sweep,
7/8, with false-alarm rate mid-pack) — revisit if Phase 3/4 integration shows this operating point
is unsuitable for soft-weighting.

## Deviations from spec + why
- **Detection built on `post_break_weight`, not raw `changepoint_probability`** — see the finding
  above. This doesn't change what Section 3 point 1 asks for (BOCPD's run-length posterior driving the
  signal); it changes which scalar summary of that posterior is used for discrete event detection.
- **No hyperparameter tuning beyond the sweep itself** — Phase 2's job per the roadmap is
  *characterization* ("sensitivity to hazard-rate prior... at least a small sweep"), not finding an
  optimal operating point. The false-alarm rate above is real and unflattering; further smoothing-
  window or event-threshold tuning is left for Phase 3/4 if the soft-weighting integration turns out
  to need it, rather than retroactively cherry-picking Phase 2 numbers to look better.
- **`run_synthetic_benchmark.py` extended in place** (per the Phase-1 report's stated intent) rather
  than adding a new entry point — Section 4 lists only three experiment scripts and none map cleanly
  to "BOCPD-only validation," so this was the least-disruptive fit. Produces its own dedicated log file
  (`phase2_bocpd_hazard_sweep_*.log`) despite sharing a script with Phase 1, per Section 12's "each
  phase's run gets its own log file."

## Pass/fail
**Pass.** Both Phase 2 acceptance criteria met: (1) BOCPD detects synthetic ground-truth breaks with
reportable lag and false-alarm rate — done, numbers above, honestly including the high false-alarm
rate; (2) sensitivity to hazard-rate prior characterized via a 6-point sweep, not a single point.

## Log files
`logs/phase2_bocpd_hazard_sweep_20260904-223346.log` (Phase 2 section); Phase 1 baselines re-run in
the same session logged separately to `logs/phase1_synthetic_baselines_20260904-223343.log`.
