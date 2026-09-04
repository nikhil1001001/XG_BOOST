# PROJECT BRIEF: Regime-Adaptive Gradient Boosting (RAGB)
### Agent execution spec — hand this file to a coding agent as standing context

---

## 0. How to use this document (read this first, agent)

You are building a research-grade ML systems project from scratch: a mixture-of-boosted-experts
architecture that combines Bayesian Online Changepoint Detection (BOCPD) with soft-weighted XGBoost
training to handle structural breaks in financial time series (fraud/credit risk).

Rules of engagement:
- Work **phase by phase**, in the order given in Section 5 (Implementation Roadmap). Do not skip
  ahead to mixture-of-experts logic before the single-expert pipeline is validated end-to-end.
- After each phase, produce a short **phase report** (what was built, what was tested, what passed/
  failed, and any deviation from spec + why) before moving to the next phase.
- Every claim of "improvement over baseline" must be backed by an actual experiment run with numbers,
  not asserted. If you can't run something (e.g. no GPU, no internet for a dataset), say so explicitly
  and substitute the nearest feasible alternative rather than silently faking results.
- Prefer readable, well-commented Python over cleverness. This is a portfolio/resume project — code
  clarity and reproducibility matter as much as results.
- Repo layout, dependencies, and acceptance criteria are specified below. Do not restructure without
  a stated reason.
- If a design decision in this brief turns out to be wrong once you're building (e.g. BOCPD library
  choice, dataset access issue), flag it, propose the fix, and proceed — don't stall waiting for
  confirmation on implementation-level details. Do stop and ask if a *scope* decision is needed
  (e.g. dropping a whole phase).
- **This is an unattended run.** No one is watching this session. There is no user to ask
  clarifying questions mid-run — if something is ambiguous, make the most reasonable decision,
  record it in the phase report, and continue. The only acceptable reasons to halt entirely are:
  (a) a scope-level decision per the line above, or (b) every item in Section 6a's fallback cascade
  has been exhausted and there is genuinely no path forward for that phase. Everything else gets a
  documented workaround, not a stop.
- Commit and push progress incrementally per Section 14 (Git & GitHub Workflow) — do not wait until
  the very end to commit. If the session is interrupted after Phase 3, the repo on GitHub should
  reflect a working Phase 3, not nothing.

---

## 0a. Environment Setup (do this before Phase 1)

Do these steps in order, verifying each before moving to the next. This section exists so the agent
never has to guess at environment state.

1. **Check Python version.** Requires Python ≥3.10. `python3 --version`. If unavailable or older,
   install via the system package manager or `pyenv` if present; don't assume a specific base image.
2. **Create and activate a virtual environment** at the repo root: `python3 -m venv .venv` then
   `source .venv/bin/activate` (or the OS-appropriate equivalent). All subsequent installs happen
   inside this venv, never system-wide.
3. **Initialize the git repo** at the root: `git init`, then create `.gitignore` immediately (see
   contents below) before anything else touches the working tree, so bulky/generated files are never
   accidentally staged.
4. **Write `pyproject.toml`** (or `requirements.txt` if the agent's tooling prefers it) pinning at
   least: `xgboost>=2.0`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `pytest`, `tqdm`,
   `openml` (for the Section 6a fallback path), `river` (for ADWIN baseline — see the flag on this
   in Section 0b), `huggingface_hub`/`datasets` (for the Lending Club fallback path). Add `kaggle`
   only if Kaggle API usage is actually attempted.
   Pin exact versions once installed successfully (`pip freeze` into a lock section) so the repo is
   reproducible, not just "latest at build time."
5. **Install and smoke-test:** `pip install -e .` (or `pip install -r requirements.txt`), then run a
   one-line `import xgboost, numpy, pandas` sanity check before writing any project code. If any
   install fails (e.g. no network for PyPI), log it clearly and try the next best alternative (e.g. a
   pre-vendored wheel, or a lighter substitute library) rather than silently proceeding without it.
6. **Confirm write access and disk space** for `data/` and `results/` — these directories can grow
   large (IEEE-CIS is several hundred MB); fail loudly and early if space is constrained rather than
   discovering it mid-Phase-5.
7. **`.gitignore` contents (create this file verbatim as a starting point, extend if needed):**
   ```
   .venv/
   __pycache__/
   *.pyc
   data/synthetic/
   data/ulb_creditcard/
   data/ieee_cis/
   data/*/
   !data/*/.gitkeep
   results/figures/*.png
   .ipynb_checkpoints/
   *.egg-info/
   .pytest_cache/
   logs/*.log
   ```
   Raw and generated data is never committed — only code, configs, `results/tables/*.csv|json`
   (small, load-bearing numeric results), and the figures actually referenced by the README (curate,
   don't dump every plot into git). Add `logs/*.log` to this file too (see Section 12) — logs are
   regenerable run artifacts, not source.
8. **First commit:** once the skeleton in Section 4 (Repo Structure) exists as empty/stub files with
   docstrings, make an initial commit ("Project skeleton + environment setup") before writing any
   real logic. This gives a clean starting point in history.

---

## 0b. Reviewer Flags (added during spec review — read before Section 1)

A few things in this brief are worth flagging explicitly rather than letting the agent silently
paper over them mid-run:

- **Section numbering gap + a bad cross-reference:** the original brief jumped from Section 11
  straight to Section 13 — there was no Section 12. This revision fills that gap with the new
  Logging & Progress Tracking section below. That gap also caused a real bug: Section 0's "commit
  and push progress incrementally per Section 13" was pointing at Testing Standards, when it clearly
  meant the Git & GitHub Workflow section (actually Section 14). Fixed in place above — worth
  flagging since a literal reading would've sent the agent to the wrong section for commit cadence.
- **`scikit-multiflow` is effectively unmaintained** (last real release years ago, doesn't reliably
  install against current numpy/scikit-learn). The original brief listed it as an equal option next
  to `river`. This revision drops it as a first-class choice — default to `river` for the ADWIN
  baseline and only reach for `scikit-multiflow` if `river`'s ADWIN implementation turns out to be
  unsuitable for some concrete reason (record that reason in the phase report if so).
- **OpenML dataset ID (`1597`) for ULB Credit Card Fraud should be verified, not trusted blindly.**
  Dataset IDs on OpenML can point to a different or deprecated version of a dataset over time. Before
  wiring this into `ulb_creditcard_loader.py`, the agent should confirm the ID resolves to the
  expected ~284k-row, `Time`+`Amount`+`V1..V28`+`Class` schema, and fall back to searching OpenML by
  name (`"creditcard"`) if it doesn't.
- **No global random-seed policy stated.** Section 5/Phase 1 requires the *synthetic generator* to be
  seeded, but nothing in the brief pins a single top-level seed (numpy, XGBoost's own `seed`/
  `random_state`, and any `river` component) that experiment scripts thread through consistently.
  Recommend adding a `--seed` CLI arg to every `run_*.py` in `experiments/`, defaulted but overridable,
  and logging the resolved seed (see Section 12) so any given results row is exactly reproducible.
- **No CI mentioned.** Section 13 requires `pytest` before every commit but nothing enforces that
  automatically. A minimal GitHub Actions workflow (`.github/workflows/tests.yml` running `pytest` on
  push) is a small addition that meaningfully strengthens the "portfolio project" framing in Section
  0 — recommended but not blocking; add it once GitHub remote push (Section 14) is working.
- **Kaggle-in-an-unattended-environment friction:** Section 6a Phase 5b targets IEEE-CIS via the
  Kaggle API, which needs `kaggle.json` credentials that plausibly won't exist in an unattended
  sandbox. This is already handled gracefully by the cascade (fall through to Elliptic, or stop at
  Lending Club) — flagging only so it's clear that Phase 5b landing on a fallback dataset is an
  *expected*, not exceptional, outcome and shouldn't be treated as a failure in the phase report. See
  Section 0c for how to get the preferred dataset instead of the fallback, if that's wanted.

---

## 0c. Practical Notes for This Specific Build

This brief will run on the user's own Windows laptop (HP Victus, Intel i7-12650H, 16GB RAM, RTX 3050
4GB VRAM, 1TB SSD) via Claude Code, not an anonymous CI sandbox. A few implications worth setting up
front so the agent doesn't rediscover them mid-run:

- **No GPU wiring needed.** XGBoost's default `hist` tree method on CPU is more than adequate at this
  project's data scale (synthetic benchmarks, ~284k-row ULB Credit Card Fraud, ~590k-row IEEE-CIS).
  Don't spend time on `device="cuda"` — it adds complexity for negligible benefit here, and 4GB VRAM
  would be a tighter constraint than the 16GB of system RAM anyway.
- **RAM budget (16GB total).** Comfortable for the synthetic benchmark and ULB Credit Card Fraud, but
  be deliberate once IEEE-CIS and the full ablation sweep (K sweep × signal choice × hard/soft
  cutover, each spawning multiple XGBoost sub-ensembles) are running concurrently or in sequence.
  Downcast dtypes on load (`float32` instead of `float64`, `category` for categorical columns), avoid
  holding multiple full copies of a wide real-data frame in memory at once, and fall back to the
  chunked-reading path from Section 12 if a single load starts to strain memory — don't discover this
  mid-Phase-5b.
- **Disk space, specifically the `C:` drive.** On Windows, `C:` is often a much smaller partition than
  total disk capacity even when the underlying SSD is large. Check free space on `C:` specifically as
  part of Section 0a step 6, not just total disk size. If `C:` is tight, point the venv, pip's cache
  (`PIP_CACHE_DIR`), and `data/` downloads at a drive with more headroom.
- **Windows shell.** Venv activation in Section 0a step 2 is `.venv\Scripts\Activate.ps1` in
  PowerShell (or `.venv\Scripts\activate.bat` in cmd), not the Unix `source .venv/bin/activate` shown
  there — use whichever matches the shell Claude Code is actually invoking commands in.
- **"Unattended" here means "don't stall on implementation-level ambiguity," not "literally
  unsupervised."** The user can be reached for scope-level decisions or to supply credentials before a
  phase that needs them. Concretely: if IEEE-CIS (the preferred Phase 5b dataset) is wanted instead of
  a Section 6a fallback, `kaggle.json` should be placed in the expected location *before* Phase 5b
  starts. If it isn't there when Phase 5b begins, proceed straight down the fallback cascade rather
  than pausing to wait for it.
- **Session length.** This build is bigger than one sitting is likely to comfortably cover. Prefer
  running it phase-by-phase across multiple Claude Code sessions using `START_PROMPT.md` /
  `RESUME_PROMPT.md` (provided alongside this brief) rather than expecting one continuous run to
  finish everything — Section 14's incremental commit/push discipline exists precisely so this is safe
  to interrupt and resume.

---

## 1. Problem Statement

Financial time series (transaction streams, credit-risk indicators) undergo structural breaks: fraud
tactics shift, macro shocks change default drivers. A gradient-boosted ensemble trained on historical
data implicitly assumes stationarity. Two standard failure modes result:

1. **Stale model** (no retraining) — silent performance decay.
2. **Naive full retrain** — discards useful old-regime signal, expensive to run continuously.

RAGB replaces the hand-tuned retrain schedule with a **mixture-of-boosted-experts controlled by an
online changepoint detector**, so the ensemble knows *when* it's likely in a new regime and *how
confident* it is, blending experts accordingly instead of a hard retrain/don't-retrain switch.

## 2. Why Standard XGBoost Is Insufficient

- Boosting rounds are computed against a fixed training distribution; there's no notion that a later
  batch of data might come from a different generative process than an earlier one.
- `xgb_model=` warm-starting lets you *add* trees, but nothing tells you *when* to start a fresh
  sub-ensemble vs. keep extending the old one.
- Splits/feature importance learned pre-break can actively mislead post-break.

## 3. Core Method

1. **Online changepoint detector (BOCPD)** runs on a monitored signal (rolling log-loss residuals of
   the current ensemble, or feature-distribution drift via sliding-window KL divergence). BOCPD
   maintains a posterior over "run length" (time since last changepoint), updated online.
2. **Mixture of boosted experts**: a small set of XGBoost sub-ensembles, one per detected regime,
   capped at K, with least-recently-relevant experts archived/pruned.
3. **Regime-posterior-weighted prediction**: blend expert outputs at inference time using the BOCPD
   run-length posterior as mixing weights.
4. **Training-time gradient blending**: weight each training instance's contribution to the *new*
   sub-ensemble's gradient/Hessian by its posterior probability of belonging to the current regime —
   soft-assigning instances near a suspected changepoint instead of a brittle hard split.

### Math core (implement exactly this)

Standard boosting round *m* fits a tree to `g_i = ∂L(y_i, F_{m-1}(x_i)) / ∂F_{m-1}(x_i)`, valid only
if the loss surface is stable. Under regime shift, pooled pre/post-break gradients are a miscalibrated
compromise between two different optima.

BOCPD's run-length posterior `P(r_t | x_{1:t})` gives, per timestep, a distribution over "steps since
last break." Use `w_i = P(instance i is post-break)` as a soft weight on leaf-level gradient/Hessian
sums: `G = Σ w_i·g_i`, `H = Σ w_i·h_i`, feeding XGBoost's leaf-weight formula `-G / (H + λ)`. This is
implemented via a **custom `obj(preds, dtrain)`** callback, not a built-in XGBoost parameter.

## 4. Repo Structure (create exactly this layout)

```
ragb/
├── README.md                      # project overview, how to reproduce every result
├── pyproject.toml                 # or requirements.txt — pin versions
├── data/
│   ├── synthetic/                 # generator output, gitignored, regenerable via script
│   ├── ulb_creditcard/            # download instructions only, not committed raw data
│   └── ieee_cis/                  # download instructions only, not committed raw data
├── src/ragb/
│   ├── __init__.py
│   ├── utils/
│   │   └── logging_config.py      # shared logger setup (see Sec 12)
│   ├── data/
│   │   ├── synthetic_generator.py # controllable regime-switching data generator
│   │   ├── real_data_loader.py    # cascading real-data acquisition (see Sec 6a)
│   │   ├── ulb_creditcard_loader.py # time-ordering + feature prep for ULB Credit Card Fraud
│   │   └── ieee_cis_loader.py     # time-ordering + feature prep for IEEE-CIS
│   ├── bocpd/
│   │   ├── detector.py            # BOCPD implementation (hazard rate, run-length posterior)
│   │   └── signals.py             # residual-loss signal, feature-drift (KL) signal
│   ├── boosting/
│   │   ├── custom_objective.py    # soft-weighted gradient/Hessian obj() for XGBoost
│   │   ├── single_expert.py       # warm-started single-ensemble pipeline (Week 3 milestone)
│   │   └── mixture.py             # expert spawn/promote/discard/prune controller (Week 4)
│   ├── baselines/
│   │   ├── static_xgb.py
│   │   ├── periodic_retrain.py
│   │   ├── sliding_window_retrain.py
│   │   └── adwin_online.py        # Idea #4 baseline — ADWIN-triggered warm-start boosting
│   ├── eval/
│   │   ├── metrics.py             # PR-AUC, detection lag, false-alarm rate, TS-AUC
│   │   └── pareto.py              # compute/accuracy Pareto frontier plotting
│   └── experiments/
│       ├── run_synthetic_benchmark.py
│       ├── run_ieee_cis_benchmark.py
│       └── run_ablations.py
├── tests/                         # pytest — unit tests per module, esp. BOCPD math + custom obj
├── notebooks/                     # exploration only, no load-bearing logic here
├── logs/                          # gitignored — one timestamped .log file per phase run (Sec 12)
└── results/
    ├── figures/
    └── tables/                    # raw experiment outputs (csv/json), not just plots
```

## 5. Implementation Roadmap (execute in this order)

### Phase 1 — Foundations
- Build `synthetic_generator.py`: simulate ≥3 distinct fraud "eras" (different generative rules),
  with known, storable ground-truth break points and a configurable hazard/switch schedule.
- Implement `static_xgb.py` and `periodic_retrain.py` baselines running end-to-end on synthetic data.
- **Acceptance criteria:** synthetic generator produces reproducible (seeded) data with recoverable
  ground truth; both baselines train and produce PR-AUC over time without error.

### Phase 2 — Changepoint detection
- Implement or adapt BOCPD (`bocpd/detector.py`) on the residual-loss signal.
- Validate detection lag and false-alarm rate **on synthetic data alone**, before touching boosting
  integration.
- **Acceptance criteria:** BOCPD detects synthetic ground-truth breaks with reportable lag and
  false-alarm rate; sensitivity to hazard-rate prior is characterized (at least a small sweep).

### Phase 3 — Soft-weighted single expert
- Implement `custom_objective.py`: soft-weighted gradient/Hessian `obj()`.
- Implement `single_expert.py`: single warm-started sub-ensemble using the soft weights end-to-end.
- **Acceptance criteria:** single-expert soft-weighted pipeline trains and evaluates on synthetic
  data; unit tests confirm the custom `obj()` reduces to standard boosting when weights are all 1.

### Phase 4 — Mixture of experts
- Implement `mixture.py`: expert spawning on posterior-shift threshold, promotion/discard logic
  (rolling validation window), recency-weighted archiving/pruning at capacity K.
- Implement posterior-weighted inference blending.
- **Acceptance criteria:** full pipeline runs end-to-end on synthetic data; spurious-expert spawn
  rate is measured and reported, not just accuracy.

### Phase 5 — Ablations + real data

Real-data acquisition is **cascading, not hard-coded** — see Section 6a for the fallback order and
implementation. Split into two sub-phases so a real-data result always exists even if the largest
dataset is unreachable:

- **Phase 5a — real-data smoke test (required, must complete):** run the full RAGB pipeline plus all
  baselines on whichever dataset `real_data_loader.py` successfully resolves first, starting with the
  smallest/most-accessible option (ULB Credit Card Fraud via OpenML, no auth needed). This validates
  the pipeline works on genuine (non-synthetic) data before committing to a larger run.
- **Phase 5b — full real-data run (best-effort, report status either way):** attempt IEEE-CIS Fraud
  (time-ordered by `TransactionDT`) via Kaggle API. If Kaggle auth is unavailable in the environment,
  fall back per the cascade in Section 6a and say explicitly in the phase report which dataset was
  actually used and why the preferred one wasn't.
- Ablations: hard vs. soft cutover; mixture-of-experts vs. single continuously-warm-started ensemble;
  K sweep (1 / 3 / unbounded); detection-signal choice (residual-loss vs. feature-drift KL).
- **Acceptance criteria:** each ablation has its own results table; Phase 5a completes on real data
  with a comparison against all baselines on the same metrics; Phase 5b's outcome (success or
  documented fallback) is reported plainly, never silently skipped.

### Phase 6 — Stress tests, writeup, resume artifacts
- Deliberately misconfigure hazard rate (too sensitive / too insensitive) and confirm expected
  false-alarm behavior appears.
- Inject gradual (non-changepoint) drift and confirm RAGB underperforms ADWIN-style baseline there —
  this is an expected, honestly-reported limitation, not a bug to hide.
- Write `README.md` with full reproduction instructions and headline results table.
- Produce final resume-bullet-ready results summary (numbers must trace to `results/tables/`).

## 6. Experiment Design

- **Synthetic regime-switching benchmark:** ground-truth breaks known by construction — the primary
  correctness check. Measure detection lag vs. ground truth.
- **Real-world test:** whichever dataset the cascade in Section 6a resolves, time-ordered — target is
  IEEE-CIS Fraud Detection, with ULB Credit Card Fraud (via OpenML, no auth) as the guaranteed-
  accessible fallback for an initial smoke test.
- **Secondary metric:** TS-AUC-style pairwise ranking evaluation alongside PR-AUC, since it measures
  whether the model's *relative* risk ranking stays regime-consistent.

## 6a. Real-Data Acquisition Cascade (implement this, don't hard-code one source)

Real-world data availability in the execution environment is not guaranteed (no login, no internet,
or a specific host being blocked). `real_data_loader.py` must try sources **in this order** and fall
through automatically, logging which one succeeded:

| Order | Dataset | Why here | Access method | Auth needed? |
|---|---|---|---|---|
| 1 | **ULB Credit Card Fraud** (`creditcard.csv`) | Real European CC transactions, ~284k rows, has a `Time` column (seconds elapsed) → directly orderable into a stream. Small, fast — good first correctness check on real data before the big run. | OpenML: `openml.datasets.get_dataset(1597)` via the `openml` Python package | No login required |
| 2 | **Lending Club Loan Data** | Real loan issue dates spanning ~2007–2018, containing a genuine macro-driven regime shift (2008 crisis is in the data) — strong fit for the credit-risk framing in Section 1. | HuggingFace Datasets hub (public, no-auth `datasets.load_dataset`) or a Kaggle mirror | No login required for the HF path |
| 3 | **IEEE-CIS Fraud Detection** (primary target dataset) | Largest, most directly comparable to the fraud framing used throughout this brief; has `TransactionAmt` and linking columns useful for future graph work (Idea #5). | Kaggle API: `kaggle competitions download -c ieee-fraud-detection` | Requires `kaggle.json` credentials |
| 4 | **Elliptic Bitcoin dataset** | Real, timestamped, illicit-transaction labels; only used as a last resort since it's a weaker fit for the credit/fraud framing than 1–3. | Kaggle or HuggingFace mirrors | Varies by mirror |

Implementation requirements for `real_data_loader.py`:
- A single entry point, e.g. `load_real_dataset(preferred=None) -> (df, metadata)`, that tries sources
  in the table order above (or starts at `preferred` if given), catches auth/network failures per
  source, and returns the first success along with `metadata["source"]` naming which one was used.
- Never fail silently: if all sources fail, raise a clear error listing what was tried and why each
  failed (missing credentials vs. no network vs. dataset removed) so the phase report can state it
  plainly rather than skipping Phase 5 without explanation.
- Cache whatever is downloaded under `data/<source_name>/` (gitignored) so repeated runs don't re-fetch.
- Each source needs its own loader module (`ulb_creditcard_loader.py`, `ieee_cis_loader.py`, etc.)
  that does time-ordering and feature prep specific to that dataset's schema — `real_data_loader.py`
  itself only handles acquisition and dispatch, not feature engineering.

## 7. Baselines (all required, not optional)

| Baseline | Purpose |
|---|---|
| Static XGBoost (trained once) | "Do nothing" floor |
| Periodic full retrain (e.g. weekly) | Common industry default |
| Naive sliding-window retrain | Stronger heuristic, still changepoint-unaware |
| ADWIN-triggered online boosting (Idea #4) | Tests whether RAGB's extra complexity (soft mixture vs. hard trigger) is actually worth it — report honestly either way |

## 8. Metrics

- PR-AUC over time (drift-adjusted)
- Detection lag (timesteps from true break to detection)
- False-alarm rate
- TS-AUC (pairwise ranking consistency)
- Compute cost (amortized training compute vs. periodic retrain)
- Backward/forward transfer if Phase 6 stress tests extend into forgetting analysis

## 9. Known Expected Failure Modes (report these, don't hide them)

- **False alarms** from a poorly tuned hazard-rate prior — fragments training data on noise.
- **Detection lag vs. false-alarm trade-off** — no free lunch; report the curve, not one cherry-picked
  operating point.
- **Expert cold-start** — a newly spawned expert overfits on little data; soft blending during
  transition is what prevents this from tanking post-break accuracy.
- **Gradual, non-changepoint drift** — a known weak spot of changepoint-based methods generally. RAGB
  is designed for discrete regime shifts; slow drift is ADWIN's (Idea #4's) territory, not RAGB's.
  State this plainly rather than over-claiming generality.

## 10. What Counts as "Genuine Improvement" (bar for claiming success)

- On synthetic data with known ground truth: PR-AUC recovers to pre-break levels **faster** than
  static or periodic-retrain baselines, at comparable total training compute.
- Report a compute/accuracy Pareto frontier — RAGB should dominate periodic full retrain (similar or
  better accuracy at lower amortized compute).
- Repeat the synthetic benchmark across multiple random regime-switch schedules; report confidence
  intervals, not a single run.

## 11. Explicit Novelty Framing (use this language in the writeup, don't overclaim)

- **Established technique:** BOCPD (Adams & MacKay, 2007); XGBoost custom objectives; warm-start
  incremental boosting.
- **Interesting combination of existing techniques:** using BOCPD's run-length posterior as a soft
  instance-weighting signal for boosting rounds.
- **Novel engineering contribution:** the promotion/discard mixture-of-experts controller plus the
  training-time soft gradient-blending mechanism as a coherent system.
- **Potentially novel research direction (be modest in interviews):** treating changepoint posterior
  as a continuous training-time reweighting signal for tree boosting, rather than a discrete retrain
  trigger — frame as "an idea I explored and validated empirically," not "a new algorithm I invented."

## 12. Logging & Progress Tracking (tqdm + structured logging)

This is an unattended, multi-hour, multi-phase run (Section 0). Bare `print()` statements are not
enough — when this session is later reviewed (by the user, or by the agent itself resuming after an
interruption), there needs to be a durable, timestamped record of what happened, in what order, and
how long each stage took, separate from the prose phase reports.

**Logging (`src/ragb/utils/logging_config.py`):**
- Use Python's standard `logging` module — no need for a heavier framework. One shared
  `get_logger(name)` helper configured once at process start, used via `logger = get_logger(__name__)`
  in every module instead of `print()`.
- Configure two handlers on the root logger: a `StreamHandler` to stdout (so progress is visible live)
  and a `FileHandler` writing to `logs/<phase>_<timestamp>.log` (so it survives after the session
  ends). Format: `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`.
- Log levels used deliberately, not everything at `INFO`:
  - `DEBUG` — per-iteration/per-batch detail (e.g. per-round boosting loss), off by default.
  - `INFO` — phase/stage start-stop, dataset shapes after loading, which real-data source resolved
    (Section 6a), hyperparameters used for a given run, final metric values.
  - `WARNING` — fallback paths taken (e.g. `river` unavailable → trying `scikit-multiflow`; preferred
    dataset unreachable → falling back per the cascade), anything a human reviewing the log should
    notice but that isn't fatal.
  - `ERROR` — a source/step failed and the code is about to fall back or halt.
- Every `experiments/run_*.py` script logs, at minimum: the resolved random seed (see Section 0b),
  the full resolved config/hyperparameters for that run, the data source actually used (for anything
  touching Section 6a), start/end wall-clock time and total duration, and the final headline metrics
  — this is what turns a log file into an audit trail for the numbers that eventually land in
  `results/tables/` and the README.
- Each phase's run gets its own log file (e.g. `logs/phase3_single_expert_20260902-141200.log`);
  don't append every phase into one giant undated log. Log files are gitignored (regenerable run
  artifacts, not source) — see Section 0a's `.gitignore`.

**Progress bars (`tqdm`):**
- Wrap any loop that runs long enough to matter for an unattended run's observability: boosting
  rounds within a single `xgb.train` custom-objective loop (if iterating manually rather than calling
  `xgb.train` directly), the outer loop over timesteps/batches in the synthetic streaming benchmark,
  the K-sweep and other ablation sweeps in `run_ablations.py`, and any real-data download/preprocessing
  step that iterates over chunks (e.g. reading IEEE-CIS in chunks if memory-constrained).
- Use `tqdm(iterable, desc="...")` with a short, specific `desc` (e.g. `"BOCPD sweep: hazard=1/200"`,
  not `"processing"`), and set `mininterval` generously (e.g. `mininterval=1.0`) so log files piped
  through `tee` or redirected to a file aren't flooded with redraw noise.
- Don't wrap trivial/fast loops (unit tests, small fixture-based tests in `tests/`, anything under
  ~1 second) — tqdm is for observability into genuinely long-running unattended work, not decoration.
- `tqdm` output goes to stdout/stderr as usual; it does not need to be duplicated into the log file
  verbatim. What *does* belong in the log file is the `logger.info(...)` line marking that stage's
  start and its completion (with count/duration), so the durable record captures the same milestones
  a human watching the tqdm bar live would have seen.

**Acceptance criteria (applies retroactively to every phase's acceptance criteria in Section 5):**
each phase's run must produce a corresponding log file under `logs/` covering that phase's execution,
and any loop over more than a trivial number of iterations must be wrapped in `tqdm`. A phase report
(Section 0) should reference its log file by name so the two artifacts stay linked.

## 13. Testing Standards

- Every module under `src/ragb/` that contains non-trivial logic (BOCPD math, custom objective,
  expert promotion/discard rules, metrics) gets a corresponding file under `tests/`.
- Priority tests, at minimum:
  - `test_custom_objective.py`: soft-weighted `obj()` reduces exactly to standard XGBoost gradient/
    Hessian when all weights = 1 (numerical equality check against `xgboost`'s built-in logloss obj).
  - `test_bocpd.py`: on a synthetic sequence with a known single changepoint, the run-length posterior
    peaks at the correct location within a small tolerance.
  - `test_synthetic_generator.py`: generator is seed-reproducible (same seed → identical data) and
    ground-truth break indices match what was actually injected.
  - `test_mixture_controller.py`: spawn/promote/discard logic behaves correctly on constructed
    posterior-sequence fixtures (not requiring a full training run).
- Run `pytest` before every commit that touches `src/`. A commit that breaks existing tests does not
  get pushed — fix first.
- Don't chase 100% coverage; prioritize correctness of the math-heavy modules over trivial getters.

## 14. Git & GitHub Workflow

Assume `git` is available and a GitHub remote needs to be created and pushed to, unattended.

1. **Repo naming:** `regime-adaptive-gradient-boosting` (or `ragb`) — lowercase, hyphenated.
2. **Remote creation:** if the GitHub CLI (`gh`) is authenticated in the environment, use
   `gh repo create <name> --public --source=. --remote=origin --push` to create and push in one step.
   If `gh` isn't authenticated but a `GITHUB_TOKEN` or equivalent credential is available in the
   environment, create the remote via the GitHub REST API (`POST /user/repos`) using that token, then
   `git remote add origin <url>` and push normally. If neither is available, complete all local work
   and commits as normal, and state clearly in the final report that GitHub push requires the user to
   run `gh repo create` (or equivalent) themselves — do not treat missing GitHub credentials as a
   reason to skip the rest of the project.
3. **Commit granularity:** one commit per meaningful unit of work, not one giant commit at the end.
   Roughly one commit per roadmap sub-step (e.g. "Add synthetic regime-switching generator",
   "Implement BOCPD run-length posterior", "Add soft-weighted custom objective + unit tests").
   Commit messages: imperative mood, present tense, no trailing period, e.g.
   `Add mixture-of-experts spawn/promote/discard controller`.
4. **Branching:** work directly on `main` for a solo project of this size — no need for a PR workflow
   against yourself. If the agent wants isolation while trying something experimental (e.g. Phase 6
   stress tests that might break things), a short-lived branch merged back is fine, but don't leave
   the repo on a stray branch at the end.
5. **Push cadence:** push to `origin` after completing each phase (Section 5), not just at the very
   end — this is what makes the "unattended run" safe per Section 0's rule about partial progress
   surviving an interruption.
6. **Tags:** after Phase 6 completes and the README is finalized, tag the commit `v1.0` 
   (`git tag -a v1.0 -m "RAGB v1.0 — full pipeline, ablations, real-data eval"` then `git push --tags`).
7. **License:** add an `MIT LICENSE` file at the root during Section 0a setup (before first commit)
   unless the user has specified otherwise — reasonable default for a portfolio project.
8. **Do not commit secrets.** Kaggle credentials, tokens, or API keys never get written into the repo
   or committed, even temporarily — they live in environment variables or a local, gitignored
   `.env`/`kaggle.json` referenced by path, never inline in code or config files that get committed.

## 15. README.md Specification

The README is the primary artifact a reviewer/interviewer will read before any code. Write it last
(Phase 6), once real results exist — do not pre-write it with placeholder numbers. Required sections,
in this order:

1. **Title + one-paragraph summary** — what RAGB is and why it exists (pull from Section 1/2, but
   condensed to 3-4 sentences, not copy-pasted wholesale).
2. **Headline results table** — the single clearest comparison (e.g. RAGB vs. static vs. periodic-
   retrain vs. ADWIN on the synthetic benchmark: PR-AUC, detection lag, compute cost), with numbers
   sourced directly from `results/tables/`. Include the real-data (Phase 5a/5b) headline result too.
3. **Method overview** — the mixture-of-experts + BOCPD architecture, one diagram (ASCII or an image
   in `results/figures/` is fine) plus 1-2 short paragraphs. Link to Section 11's novelty framing
   language almost verbatim — established technique vs. novel combination vs. engineering
   contribution — so a reader calibrates expectations correctly.
4. **Repro instructions** — exact commands from a clean clone to a finished experiment: venv setup,
   install, `python -m ragb.experiments.run_synthetic_benchmark`, etc. This must actually work if
   someone follows it; the agent should dry-run these exact commands itself before writing them down.
5. **Repo structure** — brief version of Section 4's tree, annotated.
6. **Ablations & failure modes** — summarize Section 9's known limitations honestly (gradual drift,
   false-alarm sensitivity) with a link/reference to the relevant results table or figure.
7. **What's established vs. novel here** — Section 11, condensed to a few bullets.
8. **Resume-bullet-ready summary** — 2-3 sentence version suitable for copy-paste elsewhere, with
   real numbers, not placeholders.
9. **License** and a short "built as part of independent research, feedback welcome" footer line —
   no more than that; keep it professional and understated.

Style constraints: no marketing language, no unverified superlatives ("state-of-the-art",
"revolutionary"). Every number in the README must trace to a file in `results/tables/`.

## 16. Deliverables Checklist (agent: track this explicitly)

- [ ] Environment set up per Section 0a: venv, pinned deps, `.gitignore`, initial commit
- [ ] Shared logging config (`utils/logging_config.py`) + per-phase log files under `logs/`, and
      `tqdm` wrapping all long-running loops, per Section 12
- [ ] Git repo initialized, remote created on GitHub, pushed (or documented as blocked on user auth)
- [ ] Reproducible synthetic regime-switching generator with seeded ground truth
- [ ] BOCPD detector validated standalone on synthetic data
- [ ] Custom soft-weighted `obj()` with unit tests proving correctness vs. standard boosting
- [ ] Single-expert soft-weighted pipeline, working end-to-end
- [ ] Full mixture-of-experts controller (spawn/promote/discard/prune)
- [ ] All 4 baselines implemented and run on the same benchmarks
- [ ] Full ablation suite (hard/soft, MoE vs. single, K sweep, signal choice)
- [ ] Real-data smoke test on ULB Credit Card Fraud (Phase 5a — must complete, no auth required)
- [ ] Real-data full run on IEEE-CIS Fraud or documented fallback per Section 6a (Phase 5b)
- [ ] Stress tests (misconfigured hazard rate, gradual drift injection)
- [ ] Test suite (Section 13) passing, run before every commit that touches `src/`
- [ ] `results/tables/` with raw numbers backing every claimed result
- [ ] `README.md` written per Section 15, with real (not placeholder) numbers
- [ ] `v1.0` tag created and pushed after README finalized
- [ ] Resume-bullet-ready summary, numbers traceable to results
