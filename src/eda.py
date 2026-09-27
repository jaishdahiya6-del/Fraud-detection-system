"""Exploratory Data Analysis helpers for fraud detection."""
import logging
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


def basic_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Return dictionary of dataset summary statistics."""
    if df is None or df.empty:
        raise ValueError("Provided DataFrame is empty or None.")
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def target_distribution(df: pd.DataFrame, target_col: str = "Class") -> Dict[str, Any]:
    """Calculate target class distribution and fraud percentage."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    counts = df[target_col].value_counts()
    total = len(df)
    fraud = int(counts.get(1, 0))
    genuine = int(counts.get(0, 0))
    return {
        "genuine": genuine,
        "fraud": fraud,
        "fraud_pct": round(fraud / total * 100, 4) if total > 0 else 0.0,
    }


def plot_class_distribution(df: pd.DataFrame, target_col: str = "Class") -> plt.Figure:
    """Plot bar chart of target class counts."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=target_col, data=df, ax=ax, palette=["#2E86AB", "#E63946"])
    ax.set_title("Fraud vs Genuine Transaction Count")
    ax.set_xlabel("Class (0 = Genuine, 1 = Fraud)")
    ax.set_ylabel("Count")
    return fig


def plot_amount_distribution(
    df: pd.DataFrame, target_col: str = "Class", amount_col: str = "Amount"
) -> plt.Figure:
    """Plot histogram comparison of genuine vs fraud transaction amounts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[df[target_col] == 0][amount_col], bins=50, ax=axes[0], color="#2E86AB")
    axes[0].set_title("Genuine Transaction Amount Distribution")
    sns.histplot(df[df[target_col] == 1][amount_col], bins=50, ax=axes[1], color="#E63946")
    axes[1].set_title("Fraud Transaction Amount Distribution")
    for a in axes:
        a.set_xlabel("Amount")
        a.set_ylabel("Frequency")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Plot correlation heatmap for numeric features."""
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    return fig


def plot_boxplot_amount(
    df: pd.DataFrame, target_col: str = "Class", amount_col: str = "Amount"
) -> plt.Figure:
    """Plot boxplot comparing transaction amounts across classes."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=target_col, y=amount_col, data=df, ax=ax, palette=["#2E86AB", "#E63946"])
    ax.set_title("Transaction Amount by Class")
    return fig
