# Phase 5b — Full Real-Data Run (Best-Effort)

## What was attempted
`python -m ragb.experiments.run_ieee_cis_benchmark --seed 42 --phase-label phase5b_real_data --preferred ieee_cis`
— starts the Section 6a cascade at IEEE-CIS (the preferred/primary target dataset), continuing forward
through the cascade (not back to ULB/Lending Club, already covered by Phase 5a) if it's unavailable.

## Outcome: fell back to Elliptic, as anticipated in Section 0b/0c

**IEEE-CIS failed immediately** with a clear, expected reason: no Kaggle API credentials present
(`ieee_cis_loader.py`'s `kaggle_credentials_available()` check found neither `KAGGLE_USERNAME`/
`KAGGLE_KEY` env vars nor a `kaggle.json` at `C:\Users\Nikhil\.kaggle\kaggle.json`). Per Section 0c,
this is an *expected*, not exceptional, outcome in this environment and the cascade proceeded
immediately rather than pausing to wait for credentials.

**Lending Club was correctly skipped** (it precedes IEEE-CIS in the Section 6a table but was already
resolved by Phase 5a's ULB result, and `preferred="ieee_cis"` starts the cascade forward from IEEE-CIS
per `real_data_loader.py`'s documented behavior — see `test_preferred_skips_earlier_sources`).

**Elliptic (fallback #4) succeeded**: 46,564 labeled rows (157,205 unlabeled rows dropped — ~77% of
the raw dataset has no ground-truth label and isn't usable for a supervised benchmark), 166 features,
illicit rate 9.76%, time-ordered by the dataset's native `time_step` (1-49).

**To get the preferred IEEE-CIS dataset instead:** per Section 0c, place a valid `kaggle.json` at
`C:\Users\Nikhil\.kaggle\kaggle.json` (or set `KAGGLE_USERNAME`/`KAGGLE_KEY`) and re-run this same
command — no code changes needed, `ieee_cis_loader.py` is fully implemented and will be used
automatically once credentials exist.

## Real run: full pipeline on Elliptic Bitcoin dataset (46,564 rows, seed=42)

`initial_train_frac=0.1` (4,656 train rows), evaluated on the remaining 41,908 rows.

| method | mean PR-AUC (eval windows) | n_retrains | total boosting rounds | train time (s) |
|---|---|---|---|---|
| static_xgb | 0.2989 | 1 | 100 | 0.38 |
| periodic_retrain | 0.6777 | 5 | 500 | 3.30 |
| sliding_window_retrain | 0.6661 | 5 | 500 | 1.47 |
| **adwin_online** | **0.7746** | 20 | 1,050 | 3.22 |
| single_expert | 0.3460 | 10 | 280 | 0.97 |
| mixture | 0.3945 | 10 | 440 | 1.11 |

Full tables: `results/tables/phase5b_real_data_summary.csv`, `..._pr_auc_over_time.csv`,
`..._mixture_spawn_log.csv`, `..._bocpd_events.json`, `..._data_source.json`. Log:
`logs/phase5b_real_data_20260904-231848.log` (43.58s total).

**Honest read, consistent with Phase 5a's pattern:** RAGB's `single_expert`/`mixture` again
underperform the hard-trigger and cumulative-retrain baselines by a wide margin, and `adwin_online`
(the hard-trigger comparison baseline Section 7 explicitly asks for) is the best performer here. This
is the **second real dataset in a row** where RAGB loses to simpler baselines, in contrast to its
competitive position on the synthetic benchmark (Phase 3/4) — a pattern worth stating plainly rather
than treating each real-data result as an isolated anomaly.

A likely contributing factor specific to this dataset, beyond Phase 5a's class-imbalance
hypothesis (Elliptic's 9.76% illicit rate is far less extreme than ULB's 0.17%, so imbalance alone
doesn't explain it here): **stream length**. Elliptic yields only **9 chunks** total at
`chunk_size=5000` — `mixture`'s `probation_window=20,000` (4 chunks) meant its one spawn trigger, at
chunk 9 (the very last chunk), never reached a promotion/discard decision before the stream ended (0
spawns resolved, 1 suppressed at the tail) — the mixture controller barely got to operate at all on a
stream this short. `single_expert`'s continuous down-weighting mechanism (Phase 3's finding: it
discards a large fraction of each chunk's effective training signal by design) similarly has very
little time to pay off before the stream ends. `adwin_online`'s hard, infrequent-but-decisive retrain
trigger (20 retrains, each incorporating a full accumulated window) is structurally better suited to a
short, high-turnover stream. **Combined with Phase 5a's finding, the emerging honest picture is: RAGB's
continuous soft-reweighting approach appears to need either a longer stream or milder class imbalance
than either real dataset tried here provided** to show the benefit it demonstrated on the synthetic
benchmark — a genuine, real limitation to carry into Phase 6's stress-test framing and the README,
not a result to omit or explain away.

## What was tested
- `pytest tests/` — still 51/51 passing (no source changes this phase, `real_data_loader`'s
  `preferred` behavior already covered by `test_preferred_skips_earlier_sources`, exercised for real
  here).
- Full run as tabulated above; the cascade's fallback path (IEEE-CIS -> Elliptic) is exactly what
  `test_cascade_falls_through_on_failure`/`test_preferred_skips_earlier_sources` predict, now
  confirmed against the real network/credential environment, not just mocks.

## Deviations from spec + why
- None beyond what Phase 5a already established (numeric-only Elliptic features, descriptive-only
  BOCPD pass on real data).

## Pass/fail
**Pass, via the documented fallback.** Phase 5b's acceptance criterion is explicit that landing on a
fallback dataset is an expected outcome to report plainly, not a failure — done here: IEEE-CIS was
attempted, failed for a clearly-stated and anticipated reason (missing Kaggle credentials), and the
cascade's fallback (Elliptic) produced a complete, real result set. The honest secondary finding (RAGB
underperforming simpler baselines on both real datasets tried) is reported per Section 9/0's rule
against hiding unflattering results.

## Log file
`logs/phase5b_real_data_20260904-231848.log`
