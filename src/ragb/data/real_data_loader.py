"""Cascading real-data acquisition entry point: load_real_dataset(preferred=None) -> (df, metadata).

Tries sources in the Section 6a order (ULB Credit Card Fraud -> Lending Club -> IEEE-CIS -> Elliptic),
catching auth/network failures per source and returning the first success. Handles acquisition and
dispatch only; per-dataset feature engineering lives in the dataset-specific loader modules.
"""
