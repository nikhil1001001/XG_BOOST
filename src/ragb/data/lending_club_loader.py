"""Time-ordering and feature prep for the Lending Club Loan Data dataset (Section 6a fallback #2).

Not one of Section 4's two named example loader modules -- see the "etc." in Section 6a's loader-
module requirement (same rationale as `elliptic_loader.py`). Source: a public, no-auth-required
HuggingFace mirror (`codesignal/lending-club-loan-accepted`) of the well-known Kaggle Lending Club
"accepted loans 2007-2018" file, spanning the 2008 financial crisis -- the genuine macro-driven
regime shift Section 6a's table cites as this source's reason for inclusion.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from ragb.utils.logging_config import get_logger

logger = get_logger(__name__)

HF_REPO_ID = "codesignal/lending-club-loan-accepted"
DATA_FILE = "accepted_2007_to_2018Q4.csv"

# Loans that were never resolved (still current/in-grace) have no observable good/bad outcome yet;
# only a subset of `loan_status` values represent a completed loan. "Charged Off" / "Default" =
# label 1 (bad outcome, our fraud/credit-risk-analogue positive class); "Fully Paid" = label 0.
BAD_STATUSES = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
GOOD_STATUSES = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}


def load_lending_club(cache_dir: str | Path = "data/lending_club") -> tuple[pd.DataFrame, pd.Series, dict]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching Lending Club Loan Data from HuggingFace hub (%s)... (~1.7GB, may take a while)", HF_REPO_ID)
    data_path = hf_hub_download(HF_REPO_ID, DATA_FILE, repo_type="dataset", local_dir=str(cache_dir))

    df = pd.read_csv(data_path, low_memory=False)
    df = df[df["loan_status"].isin(BAD_STATUSES | GOOD_STATUSES)].copy()
    df["label"] = df["loan_status"].isin(BAD_STATUSES).astype(int)

    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df = df.dropna(subset=["issue_d"]).sort_values("issue_d", kind="stable").reset_index(drop=True)

    # Numeric columns only: XGBoost's default (non-categorical) training path needs int/float/bool
    # dtypes, and Lending Club's ~30 object-dtype columns are mostly high-cardinality free text
    # (url, desc, title, emp_title, zip_code) or redundant status strings -- not worth the added
    # complexity of categorical encoding for a fallback data source in a regime-shift benchmark
    # where the numeric financial fields (loan amount, income, DTI, FICO range, rates, ...) already
    # carry the bulk of the credit-risk signal. A future pass could add categorical support if this
    # source becomes the primary real-data result rather than a fallback.
    drop_cols = {"loan_status", "label", "issue_d"}
    feature_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].astype("float32")
    y = df["label"]

    metadata = {
        "source": "lending_club",
        "n_rows": len(df),
        "n_features": X.shape[1],
        "bad_loan_rate": float(y.mean()),
        "time_column": "issue_d",
        "date_range": [str(df["issue_d"].min()), str(df["issue_d"].max())],
    }
    logger.info(
        "Lending Club loaded: %d rows (%s to %s), %d features, bad_loan_rate=%.4f",
        metadata["n_rows"], *metadata["date_range"], metadata["n_features"], metadata["bad_loan_rate"],
    )
    return X, y, metadata
