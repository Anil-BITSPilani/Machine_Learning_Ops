import io
import logging
import os
import time
import numpy as np
import pandas as pd
import requests
from sklearn.datasets import fetch_openml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data_pipeline")

FEATURE_COLS = ["Amount", "hour_sin", "hour_cos"] + [f"V{i}" for i in range(1, 29)]
TARGET_COL = "Class"
OPENML_DATASET_ID = 42175
OPENML_PARQUET_URL = "https://openml.org"


def _download_parquet_with_retries(url: str, max_retries: int = 3) -> pd.DataFrame:
    """Download pre-built OpenML Parquet mirror with retry logic to avoid chunked ARFF drops."""
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Downloading Credit Card Fraud Parquet from OpenML (attempt %d/%d)...", attempt, max_retries)
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            return pd.read_parquet(io.BytesIO(resp.content))
        except Exception as exc:
            logger.warning("Download attempt %d failed (%s: %s)", attempt, type(exc).__name__, exc)
            if attempt < max_retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download dataset after {max_retries} attempts from {url}")


def fetch_and_cache_data(data_path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Download 31-column Credit Card Fraud dataset (with Time) and cache locally."""
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    if os.path.exists(data_path):
        df_cached = pd.read_csv(data_path)
        if "Time" in df_cached.columns and TARGET_COL in df_cached.columns:
            logger.info("Loaded cached dataset from %s with shape %s", data_path, df_cached.shape)
            return df_cached
        logger.warning(
            "Cached dataset at %s is missing 'Time' column (shape=%s); re-fetching full dataset...",
            data_path,
            df_cached.shape,
        )

    try:
        df = _download_parquet_with_retries(OPENML_PARQUET_URL)
    except Exception as exc:
        logger.warning("Parquet mirror unavailable (%s); falling back to sklearn fetch_openml...", exc)
        dataset = fetch_openml(data_id=OPENML_DATASET_ID, as_frame=True, parser="auto", n_retries=5, delay=2.0)
        df = dataset.frame.copy()

    if "Time" not in df.columns:
        raise RuntimeError(f"Expected 'Time' column in dataset, got {list(df.columns)}")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL]).astype(int)
    df.to_csv(data_path, index=False)
    logger.info("Saved full 31-column dataset to %s with shape %s", data_path, df.shape)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create cyclical 24-hour features from elapsed seconds without look-ahead leakage."""
    out = df.copy()
    hour_of_day = (out["Time"].astype(float) / 3600.0) % 24.0
    out["hour_sin"] = np.sin(2.0 * np.pi * hour_of_day / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour_of_day / 24.0)
    return out


def temporal_train_val_test_split(
    df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15
):
    """Enforce strict chronological ordering (70% Train / 15% Val / 15% Test)."""
    df_sorted = df.sort_values("Time").reset_index(drop=True)
    df_feat = add_temporal_features(df_sorted)

    n = len(df_feat)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_feat.iloc[:train_end].copy()
    val_df = df_feat.iloc[train_end:val_end].copy()
    test_df = df_feat.iloc[val_end:].copy()

    return (
        (train_df[FEATURE_COLS], train_df[TARGET_COL]),
        (val_df[FEATURE_COLS], val_df[TARGET_COL]),
        (test_df[FEATURE_COLS], test_df[TARGET_COL]),
    )


if __name__ == "__main__":
    raw_df = fetch_and_cache_data()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = temporal_train_val_test_split(raw_df)
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        print(f"{name:5s}: {len(y):6d} rows | Frauds: {y.sum():3d} ({y.mean()*100:.3f}%)")