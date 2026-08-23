"""Exploratory Data Analysis helpers for fraud detection."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")


def basic_info(df: pd.DataFrame) -> dict:
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def target_distribution(df: pd.DataFrame, target_col="Class") -> dict:
    counts = df[target_col].value_counts()
    total = len(df)
    fraud = int(counts.get(1, 0))
    genuine = int(counts.get(0, 0))
    return {
        "genuine": genuine,
        "fraud": fraud,
        "fraud_pct": round(fraud / total * 100, 4),
    }


def plot_class_distribution(df, target_col="Class"):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=target_col, data=df, ax=ax)
    ax.set_title("Fraud vs Genuine Transaction Count")
    ax.set_xlabel("Class (0 = Genuine, 1 = Fraud)")
    ax.set_ylabel("Count")
    return fig


def plot_amount_distribution(df, target_col="Class", amount_col="Amount"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[df[target_col] == 0][amount_col], bins=50, ax=axes[0], color="steelblue")
    axes[0].set_title("Genuine Transaction Amount Distribution")
    sns.histplot(df[df[target_col] == 1][amount_col], bins=50, ax=axes[1], color="crimson")
    axes[1].set_title("Fraud Transaction Amount Distribution")
    for a in axes:
        a.set_xlabel("Amount")
        a.set_ylabel("Frequency")
    return fig


def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    return fig


def plot_boxplot_amount(df, target_col="Class", amount_col="Amount"):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=target_col, y=amount_col, data=df, ax=ax)
    ax.set_title("Transaction Amount by Class")
    return fig
