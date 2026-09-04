"""Time-ordering and feature prep for the Elliptic Bitcoin dataset (Section 6a fallback #4).

Not listed as one of Section 4's two named example loader modules, but Section 6a explicitly says
"each source needs its own loader module (ulb_creditcard_loader.py, ieee_cis_loader.py, etc.)" --
the "etc." covers this and `lending_club_loader.py`. Source: a public, no-auth-required HuggingFace
mirror (`yhoma/elliptic-bitcoin-dataset`) of the original Elliptic dataset (Weber et al. 2019).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from ragb.utils.logging_config import get_logger

logger = get_logger(__name__)

HF_REPO_ID = "yhoma/elliptic-bitcoin-dataset"
FEATURES_FILE = "elliptic_txs_features.csv"
CLASSES_FILE = "elliptic_txs_classes.csv"


def load_elliptic(cache_dir: str | Path = "data/elliptic") -> tuple[pd.DataFrame, pd.Series, dict]:
    """The features file has no header: column 0 is txId, column 1 is the time step (1..49, the
    dataset's native temporal ordering), columns 2+ are 165 anonymized local + aggregated features.
    The classes file labels each txId '1' (illicit), '2' (licit), or 'unknown' (unlabeled, ~77% of
    rows) -- unlabeled rows are dropped since they carry no ground truth for a supervised benchmark.
    Sorted by time step (ties broken by txId for a stable, reproducible order) to give a genuine
    time-ordered real-world stream, per Section 6a.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching Elliptic Bitcoin dataset from HuggingFace hub (%s)...", HF_REPO_ID)
    features_path = hf_hub_download(HF_REPO_ID, FEATURES_FILE, repo_type="dataset", local_dir=str(cache_dir))
    classes_path = hf_hub_download(HF_REPO_ID, CLASSES_FILE, repo_type="dataset", local_dir=str(cache_dir))

    feature_cols = ["txId", "time_step"] + [f"f{i}" for i in range(165)]
    features = pd.read_csv(features_path, header=None, names=feature_cols)
    classes = pd.read_csv(classes_path)

    df = features.merge(classes, on="txId", how="inner")
    df = df[df["class"] != "unknown"].copy()
    if df.empty:
        raise ValueError("Elliptic dataset merge produced no labeled rows -- schema may have changed")

    df["label"] = (df["class"] == "1").astype(int)  # '1' = illicit (fraud-analogue), '2' = licit
    df = df.sort_values(["time_step", "txId"], kind="stable").reset_index(drop=True)

    y = df["label"]
    X = df[[c for c in feature_cols if c != "txId"]].astype("float32")

    metadata = {
        "source": "elliptic",
        "n_rows": len(df),
        "n_features": X.shape[1],
        "fraud_rate": float(y.mean()),
        "time_column": "time_step",
        "n_unlabeled_dropped": len(features) - len(df),
    }
    logger.info(
        "Elliptic Bitcoin dataset loaded: %d labeled rows (%d unlabeled dropped), %d features, illicit_rate=%.4f",
        metadata["n_rows"], metadata["n_unlabeled_dropped"], metadata["n_features"], metadata["fraud_rate"],
    )
    return X, y, metadata
