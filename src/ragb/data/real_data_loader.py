"""Cascading real-data acquisition entry point: load_real_dataset(preferred=None) -> (df, metadata).

Tries sources in the Section 6a order (ULB Credit Card Fraud -> Lending Club -> IEEE-CIS -> Elliptic),
catching auth/network failures per source and returning the first success. Handles acquisition and
dispatch only; per-dataset feature engineering lives in the dataset-specific loader modules.
"""

from __future__ import annotations

import pandas as pd

from ragb.data.elliptic_loader import load_elliptic
from ragb.data.ieee_cis_loader import load_ieee_cis
from ragb.data.lending_club_loader import load_lending_club
from ragb.data.ulb_creditcard_loader import load_ulb_creditcard
from ragb.utils.logging_config import get_logger

logger = get_logger(__name__)

# Section 6a table order.
CASCADE = [
    ("ulb_creditcard", load_ulb_creditcard),
    ("lending_club", load_lending_club),
    ("ieee_cis", load_ieee_cis),
    ("elliptic", load_elliptic),
]


class RealDataUnavailableError(RuntimeError):
    """Raised when every source in the Section 6a cascade failed. Never fail silently: the message
    lists exactly what was tried and why each failed (per Section 6a's requirement), so a phase
    report can state the outcome plainly rather than skip Phase 5 without explanation.
    """


def load_real_dataset(preferred: str | None = None) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Returns (X, y, metadata) from the first cascade source that succeeds, with
    metadata["source"] naming which one was used. If `preferred` is given, the cascade starts at
    that source and continues forward through the remaining (not-yet-tried) sources in table order
    -- it does not wrap back to sources earlier in the table, since those are assumed already
    covered by an earlier call (e.g. Phase 5a already resolved ulb_creditcard; Phase 5b's
    preferred="ieee_cis" call shouldn't silently re-resolve back to it).
    """
    names = [name for name, _ in CASCADE]
    start_idx = 0
    if preferred is not None:
        if preferred not in names:
            raise ValueError(f"Unknown preferred source '{preferred}'; must be one of {names}")
        start_idx = names.index(preferred)

    attempted = []
    for name, loader_fn in CASCADE[start_idx:]:
        logger.info("real_data_loader: attempting source '%s'...", name)
        try:
            X, y, metadata = loader_fn()
            logger.info("real_data_loader: source '%s' succeeded (%d rows)", name, len(X))
            return X, y, metadata
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            attempted.append((name, reason))
            logger.warning("real_data_loader: source '%s' failed, falling back. Reason: %s", name, reason)

    detail = "\n".join(f"  - {name}: {reason}" for name, reason in attempted)
    raise RealDataUnavailableError(
        f"All real-data sources in the Section 6a cascade failed (tried {[a[0] for a in attempted]}):\n{detail}"
    )
