"""Time-ordering and feature prep for the ULB Credit Card Fraud dataset (OpenML id 1597, ~284k rows)."""

from __future__ import annotations

from pathlib import Path

import openml
import pandas as pd

from ragb.utils.logging_config import get_logger

logger = get_logger(__name__)

OPENML_DATASET_ID = 1597
EXPECTED_N_ROWS_APPROX = 284_807
EXPECTED_COLUMNS = {"Time", "Amount", "Class", *(f"V{i}" for i in range(1, 29))}


def _read_cached_parquet(cache_dir: Path) -> pd.DataFrame:
    matches = list(cache_dir.glob(f"**/dataset_{OPENML_DATASET_ID}.pq"))
    if not matches:
        raise FileNotFoundError(f"Could not locate cached dataset_{OPENML_DATASET_ID}.pq under {cache_dir}")
    return pd.read_parquet(matches[0])


def load_ulb_creditcard(cache_dir: str | Path = "data/ulb_creditcard") -> tuple[pd.DataFrame, pd.Series, dict]:
    """Downloads (or loads from OpenML's own cache) the ULB Credit Card Fraud dataset, verifies its
    schema matches what Section 0b's review flagged as needing confirmation (not just trusting the
    dataset id blindly), time-orders it by the `Time` column (seconds elapsed since the first
    transaction -- already monotonic in this dataset, but sorted explicitly rather than assumed), and
    returns (X, y, metadata).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # openml's own cache dir is process-wide config, not per-call; point it at our gitignored cache
    # directory so repeated runs don't re-download (Section 6a requirement).
    openml.config.set_root_cache_directory(str(cache_dir.resolve()))

    logger.info("Fetching ULB Credit Card Fraud dataset from OpenML (id=%d)...", OPENML_DATASET_ID)
    dataset = openml.datasets.get_dataset(OPENML_DATASET_ID, download_data=True)

    actual_columns = {f.name for f in dataset.features.values()}
    if not EXPECTED_COLUMNS.issubset(actual_columns):
        missing = EXPECTED_COLUMNS - actual_columns
        raise ValueError(
            f"OpenML dataset {OPENML_DATASET_ID} does not have the expected ULB Credit Card Fraud "
            f"schema (missing columns: {missing}). Per Section 0b, this ID should be re-verified or "
            f"the loader should fall back to searching OpenML by name ('creditcard')."
        )

    # NOTE: openml's own `get_data()` silently drops the `Time` column -- it's registered as this
    # dataset's `row_id_attribute` on OpenML, which get_data() always excludes from the returned
    # frame regardless of the `target=` argument (confirmed: still missing even calling get_data()
    # with no target at all). Since `Time` is exactly the column this loader needs for time-ordering,
    # read the cached raw parquet file directly instead of going through get_data().
    df = pd.read_parquet(dataset.data_file) if dataset.data_file else _read_cached_parquet(cache_dir)
    if len(df) < EXPECTED_N_ROWS_APPROX * 0.9:
        raise ValueError(f"OpenML dataset {OPENML_DATASET_ID} returned {len(df)} rows, expected ~{EXPECTED_N_ROWS_APPROX}")

    df = df.sort_values("Time", kind="stable").reset_index(drop=True)

    y = df["Class"].astype(int)
    y.name = "label"
    feature_cols = [c for c in df.columns if c not in ("Class",)]
    X = df[feature_cols].astype("float32")

    metadata = {
        "source": "ulb_creditcard",
        "openml_dataset_id": OPENML_DATASET_ID,
        "n_rows": len(df),
        "n_features": X.shape[1],
        "fraud_rate": float(y.mean()),
        "time_column": "Time",
    }
    logger.info(
        "ULB Credit Card Fraud loaded: %d rows, %d features, fraud_rate=%.5f",
        metadata["n_rows"], metadata["n_features"], metadata["fraud_rate"],
    )
    return X, y, metadata
