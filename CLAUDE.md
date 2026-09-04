# RAGB — Regime-Adaptive Gradient Boosting

Full spec: `RAGB_project_brief.md` in this repo root — read it in full before doing any work here.
This file is a quick-reference summary of the rules that matter every session; the brief is
authoritative if anything here seems to conflict with it.

## What this project is
A mixture-of-boosted-experts system: Bayesian Online Changepoint Detection (BOCPD) driving
soft-weighted XGBoost training, to handle structural breaks in financial time series (fraud/credit
risk). Full method in Section 3 of the brief.

## Non-negotiable rules
- Work phase-by-phase, in the order given in Section 5. Don't skip ahead to mixture-of-experts logic
  before the single-expert pipeline is validated end-to-end.
- After every phase: a short phase report (what was built/tested, pass/fail, deviations + why) before
  moving on.
- Every "improvement over baseline" claim needs an actual experiment run with numbers — never assert
  results. If something can't be run (no GPU, no network for a dataset), say so and substitute the
  nearest feasible alternative rather than faking it.
- Run `pytest` before every commit that touches `src/`; a commit that breaks tests doesn't get pushed.
- Commit granularity: one commit per roadmap sub-step, imperative-mood messages. Push to `origin`
  after each phase completes (Section 14) — don't wait until the end.
- Every experiment run gets its own timestamped log file under `logs/` (Section 12), and any
  non-trivial loop is wrapped in `tqdm`.
- `README.md` (Section 15) is written last, once real results exist — no placeholder numbers, ever.
  Every number in it must trace to a file in `results/tables/`.

## This machine
Windows, HP Victus, i7-12650H, 16GB RAM, RTX 3050 (4GB VRAM), 1TB SSD — check `C:` free space
specifically, it's usually a smaller partition than the whole drive. No GPU wiring needed; CPU
XGBoost `hist` is fine at this project's data scale. Watch RAM during the full IEEE-CIS run and the
ablation sweep; downcast dtypes. Full detail in Section 0c of the brief.

## Ambiguity / when to stop
Make the most reasonable call on any implementation-level ambiguity, record it in the phase report,
and keep going — there's no one watching turn-by-turn. Only stop for a genuine scope-level decision
(e.g. dropping a whole phase) or when Section 6a's entire fallback cascade is exhausted with no path
forward for that phase.

## Session handling
This build is bigger than one sitting. Use `START_PROMPT.md` to kick off the first session, and
`RESUME_PROMPT.md` at the start of every session after that — it has the agent reconstruct where the
build left off (git log, phase reports, `results/tables/`) before continuing, rather than restarting
or redoing finished phases.
