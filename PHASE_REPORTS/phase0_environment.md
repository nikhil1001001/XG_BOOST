# Phase 0 — Environment Setup (Section 0a)

## What was built
- Confirmed Python ≥3.10 available (anaconda `python` = 3.11.7, msys64 `python3` = 3.12.10; chose
  3.11.7 for the venv — first on PATH, no reason to prefer the other).
- Created venv, initialized git repo (`main` branch), wrote `.gitignore`, created the full Section 4
  repo skeleton (all module stub files with one-line docstrings), `pyproject.toml`, MIT `LICENSE`,
  and a placeholder `README.md` (no numbers — real README is written in Phase 6 per Section 15).
- Installed all pinned dependencies from Section 0a step 4 (`xgboost`, `numpy`, `pandas`,
  `scikit-learn`, `scipy`, `matplotlib`, `pytest`, `tqdm`, `openml`, `river`, `huggingface_hub`,
  `datasets`) via `pip install -e ".[dev]"`. `kaggle` extra deliberately not installed yet — will add
  only if/when Phase 5b actually attempts the Kaggle path (Section 0a step 4 says "only if actually
  attempted").
- Ran the import smoke test (`import xgboost, numpy, pandas, sklearn, scipy, matplotlib, river,
  openml, datasets, huggingface_hub`) — passed. Versions: xgboost 3.2.0, numpy 2.4.6, pandas 3.0.5,
  scikit-learn 1.9.0, scipy 1.17.1, river 0.26.1.
- Pinned exact versions to `requirements-lock.txt` via `pip freeze` (63 packages incl. transitive
  deps).

## Deviation from spec + why (implementation-level decisions, made and logged per Section 0's rule)

1. **Venv, pip cache, and `data/` moved off the `C:` drive.** `C:` has only 6.2 GB free (98% full,
   288.8 GB used of ~295 GB) — confirmed via `Get-PSDrive` before touching anything, per Section 0c's
   explicit instruction to check `C:` specifically. `E:` has 83.3 GB free. Given IEEE-CIS alone is
   several hundred MB and the venv + pip cache for this dependency set is on the order of 1-2 GB, 6.2
   GB free was judged too tight to safely run the full roadmap (ablation sweeps, real-data downloads)
   without risking a mid-run disk-full failure. Concretely:
   - venv lives at `E:\ragb_env\.venv` (not `.venv/` inside the repo). Repo root still has `.venv/` in
     `.gitignore` in case a local one is ever created instead.
   - `PIP_CACHE_DIR` set to `E:\pip_cache` for all installs (exported inline per command; not a
     permanent env var, so future sessions need to set it again before `pip install`).
   - `data/synthetic/`, `data/ulb_creditcard/`, `data/ieee_cis/` are Windows directory **junctions**
     (`mklink /J`, no admin rights required, unlike symlinks) pointing at
     `E:\ragb_data\{synthetic,ulb_creditcard,ieee_cis}\`. This keeps the repo-relative path
     (`data/synthetic/...`) working unchanged for all code, while the actual bytes land on `E:`.
     Verified writable through the junction (`data/synthetic/.gitkeep` written and read back
     successfully). `results/` (small — tables/figures only) stays on `C:` inside the repo as specced.
   - **For any future session/agent picking this up:** before running `pip install` or any data
     download step, activate the venv via `E:\ragb_env\.venv\Scripts\Activate.ps1` (PowerShell) and
     set `$env:PIP_CACHE_DIR = "E:\pip_cache"` if installing anything new. The `data/*` junctions are
     already in place on disk and don't need to be recreated unless the repo is re-cloned fresh (see
     note below).
   - **Known limitation:** junctions are local-machine filesystem state, not something git can
     recreate on a fresh clone. If this repo is cloned onto another machine, `data/synthetic/` etc.
     will just be plain empty directories there (which is actually fine — they're gitignored content
     directories the code regenerates/redownloads into on first run; nothing load-bearing depends on
     the junction itself, only on the path existing and being writable).
2. **Git default branch renamed to `main`** immediately after `git init` (default was `master`) to
   match the GitHub remote convention used in Section 14.
3. **Added a `PHASE_REPORTS/` directory** (not in Section 4's tree) to hold this and subsequent phase
   reports as separate files rather than one growing doc — additive, doesn't restructure anything
   Section 4 specified.
4. **`kaggle` extra left uninstalled** for now, per Section 0a step 4's own instruction ("Add `kaggle`
   only if Kaggle API usage is actually attempted") — will install when/if Phase 5b reaches that point.
5. **Fixed a real bug in the brief's own `.gitignore` template (Section 0a step 7).** The specified
   pattern excludes `data/synthetic/`, `data/ulb_creditcard/`, `data/ieee_cis/` (the directories
   themselves) and separately `data/*/`, then tries `!data/*/.gitkeep` to re-include the keep files.
   This doesn't work: git never traverses into a directory that's itself excluded, so a negation on a
   path inside it is silently ignored — confirmed empirically (`git add -A` staged zero files under
   `data/` with the brief's literal pattern). Fixed by excluding directory *contents*
   (`data/<name>/*`) instead of the directories, with one `!data/<name>/.gitkeep` negation per
   directory — verified this correctly stages the three `.gitkeep` files while still ignoring any real
   downloaded data placed in those directories later.

## What was tested
- `python -m venv` succeeded, venv Python confirmed at 3.11.7.
- `pip install -e ".[dev]"` completed with no errors; full dependency tree resolved and installed
  (xgboost 3.2.0 up through pytest 9.1.1 — see `requirements-lock.txt` for the complete pinned list).
- Import smoke test for every Section 0a step-4 package: passed.
- Write-access check on `data/synthetic/` (through the junction) and `results/tables/`: passed.

## Pass/fail
All Section 0a steps (1 through 8, minus step 8's actual commit which follows this report) completed
successfully. No blockers encountered. No fallback cascade needed at this stage.

## Log file
No `logs/` entry for this stage — Section 12's logging config doesn't exist as runnable code yet
(it's a docstring stub created in this phase's skeleton, implemented properly starting Phase 1 setup
if needed, or first real use in Phase 1). Environment setup has no experiment run to log; this phase
report is the durable record for this stage instead.
