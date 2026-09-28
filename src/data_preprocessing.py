"""Data loading, cleaning, validation, and preprocessing for fraud detection."""
import os
import logging
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

TARGET_COL = "Class"


def load_data(path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Load dataset from CSV file path with validation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at path: {path}")

    logger.info(f"Loading data from {path}...")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Loaded dataset from {path} is empty.")

    logger.info(f"Successfully loaded dataset with shape {df.shape}")
    return df


def validate_data(df: pd.DataFrame, target_col: str = TARGET_COL) -> bool:
    """Validate dataset structure and required columns."""
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame columns: {list(df.columns)}")

    unique_targets = df[target_col].dropna().unique()
    if not set(unique_targets).issubset({0, 1}):
        raise ValueError(f"Target column '{target_col}' must contain binary values (0 or 1). Found: {unique_targets}")

    return True


def clean_data(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Clean dataset by removing duplicates and missing values."""
    validate_data(df, target_col)

    initial_rows = len(df)
    df_cleaned = df.drop_duplicates()
    dups_removed = initial_rows - len(df_cleaned)

    initial_cleaned = len(df_cleaned)
    df_cleaned = df_cleaned.dropna()
    nans_removed = initial_cleaned - len(df_cleaned)

    if dups_removed > 0 or nans_removed > 0:
        logger.info(f"Cleaned data: removed {dups_removed} duplicates and {nans_removed} NaN rows.")

    return df_cleaned.reset_index(drop=True)


def split_features_target(
    df: pd.DataFrame, target_col: str = TARGET_COL
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features X and target Series y."""
    validate_data(df, target_col)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform stratified split to maintain class ratio between train and test splits."""
    if len(X) != len(y):
        raise ValueError(f"X and y must have the same length. Got {len(X)} and {len(y)}.")

    logger.info(f"Performing stratified split (test_size={test_size}, random_state={random_state})...")
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cols_to_scale: Tuple[str, ...] = ("Amount", "Time")
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit scaler on train set only, then transform both train and test to prevent data leakage."""
    scaler = StandardScaler()
    cols = [c for c in cols_to_scale if c in X_train.columns]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    if cols:
        logger.info(f"Scaling features: {cols}")
        X_train_scaled[cols] = scaler.fit_transform(X_train[cols])
        X_test_scaled[cols] = scaler.transform(X_test[cols])
    else:
        logger.warning(f"None of specified scale columns {cols_to_scale} found in DataFrame.")
        scaler.fit(X_train)  # fallback fit

    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(
    path: str = "data/creditcard.csv",
    target_col: str = TARGET_COL
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Run full end-to-end data loading, cleaning, splitting, and scaling pipeline."""
    df = load_data(path)
    df = clean_data(df, target_col=target_col)
    X, y = split_features_target(df, target_col=target_col)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    X_train, X_test, scaler = scale_features(X_train, X_test)
    return X_train, X_test, y_train, y_test, scaler
