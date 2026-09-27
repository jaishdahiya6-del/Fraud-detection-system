"""Unit tests for src.data_preprocessing module."""
import os
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data_preprocessing import (
    load_data,
    validate_data,
    clean_data,
    split_features_target,
    stratified_split,
    scale_features,
    preprocess_pipeline,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(np.random.randn(n, 5), columns=[f"V{i}" for i in range(1, 6)])
    df["Time"] = np.random.randint(0, 1000, n)
    df["Amount"] = np.random.exponential(50, n)
    df["Class"] = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
    return df


def test_validate_data_valid(sample_df):
    assert validate_data(sample_df) is True


def test_validate_data_missing_target(sample_df):
    df_no_target = sample_df.drop(columns=["Class"])
    with pytest.raises(ValueError, match="Target column 'Class' not found"):
        validate_data(df_no_target)


def test_validate_data_non_binary_target(sample_df):
    sample_df["Class"] = 5
    with pytest.raises(ValueError, match="must contain binary values"):
        validate_data(sample_df)


def test_clean_data(sample_df):
    # Add duplicate row and NaN
    dup = sample_df.iloc[[0]].copy()
    sample_df = pd.concat([sample_df, dup], ignore_index=True)
    sample_df.loc[1, "V1"] = np.nan

    cleaned = clean_data(sample_df)
    assert len(cleaned) < len(sample_df)
    assert cleaned.isnull().sum().sum() == 0


def test_split_features_target(sample_df):
    X, y = split_features_target(sample_df)
    assert "Class" not in X.columns
    assert len(X) == len(y)
    assert y.name == "Class"


def test_stratified_split(sample_df):
    X, y = split_features_target(sample_df)
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, random_state=42)
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.1)


def test_scale_features(sample_df):
    X, y = split_features_target(sample_df)
    X_train, X_test, _, _ = stratified_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, cols_to_scale=("Amount", "Time"))

    assert isinstance(scaler, StandardScaler)
    assert X_train_scaled["Amount"].mean() == pytest.approx(0.0, abs=1e-5)
    assert X_train_scaled["Amount"].std() == pytest.approx(1.0, abs=1e-1)
