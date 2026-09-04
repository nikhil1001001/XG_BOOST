"""Time-ordering (by TransactionDT) and feature prep for the IEEE-CIS Fraud Detection dataset."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pandas as pd

from ragb.utils.logging_config import get_logger

logger = get_logger(__name__)

KAGGLE_COMPETITION = "ieee-fraud-detection"


def _kaggle_credentials_path() -> Path:
    return Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"


def kaggle_credentials_available() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return _kaggle_credentials_path().exists()


def load_ieee_cis(cache_dir: str | Path = "data/ieee_cis") -> tuple[pd.DataFrame, pd.Series, dict]:
    """Requires Kaggle API credentials (Section 0c/0b: plausibly absent in an unattended sandbox --
    this is expected, not exceptional; `real_data_loader.py` catches the resulting error and falls
    through the Section 6a cascade). Raises FileNotFoundError immediately, before attempting any
    network call, if no credentials are found, so the cascade fails fast with a clear reason.
    """
    if not kaggle_credentials_available():
        raise FileNotFoundError(
            f"Kaggle API credentials not found (no KAGGLE_USERNAME/KAGGLE_KEY env vars, no kaggle.json "
            f"at {_kaggle_credentials_path()}). See https://www.kaggle.com/docs/api to obtain one, or "
            "let real_data_loader.py fall back to the next Section 6a source."
        )

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    import kaggle  # imported lazily: only needed (and only required to be installed) on this path
    kaggle.api.authenticate()

    train_transaction_path = cache_dir / "train_transaction.csv"
    if not train_transaction_path.exists():
        logger.info("Downloading IEEE-CIS Fraud Detection competition data from Kaggle...")
        kaggle.api.competition_download_files(KAGGLE_COMPETITION, path=str(cache_dir))
        zip_path = cache_dir / f"{KAGGLE_COMPETITION}.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(cache_dir)

    if not train_transaction_path.exists():
        raise FileNotFoundError(f"Expected {train_transaction_path} after Kaggle download but it's missing")

    df = pd.read_csv(train_transaction_path)
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)

    y = df["isFraud"].astype(int)
    y.name = "label"
    drop_cols = {"isFraud", "TransactionID"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category")
        elif pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("float32")

    metadata = {
        "source": "ieee_cis",
        "n_rows": len(df),
        "n_features": X.shape[1],
        "fraud_rate": float(y.mean()),
        "time_column": "TransactionDT",
    }
    logger.info(
        "IEEE-CIS Fraud Detection loaded: %d rows, %d features, fraud_rate=%.5f",
        metadata["n_rows"], metadata["n_features"], metadata["fraud_rate"],
    )
    return X, y, metadata
