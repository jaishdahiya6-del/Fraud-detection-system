"""Evaluation metrics, confusion matrix, threshold optimization, and feature importance."""
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve,
)

logger = logging.getLogger(__name__)


def get_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Extract class 1 prediction probabilities or decision values from model."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] > 1:
            return probs[:, 1]
        return probs.ravel()
    elif hasattr(model, "decision_function"):
        return model.decision_function(X)
    else:
        raise AttributeError("Model does not implement `predict_proba` or `decision_function`.")


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> Dict[str, float]:
    """Calculate classification metrics for model at given threshold."""
    y_prob = get_probabilities(model, X_test)
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate ROC-AUC and PR-AUC safely
    try:
        roc_auc = float(roc_auc_score(y_test, y_prob))
    except Exception as e:
        logger.warning(f"Failed to compute ROC-AUC: {e}")
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_test, y_prob))
    except Exception as e:
        logger.warning(f"Failed to compute PR-AUC: {e}")
        pr_auc = 0.0

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def get_confusion_matrix(model: Any, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> Dict[str, Any]:
    """Compute confusion matrix and return breakdown dict."""
    y_prob = get_probabilities(model, X_test)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Fallback for single class case in y_test/pred
        tn = int(cm[0, 0]) if cm.shape[0] > 0 else 0
        fp = 0
        fn = 0
        tp = 0
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "matrix": cm}


def plot_confusion_matrix(cm: np.ndarray) -> plt.Figure:
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Genuine", "Fraud"], yticklabels=["Genuine", "Fraud"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_pr_curve(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> plt.Figure:
    """Plot Precision-Recall curve."""
    y_prob = get_probabilities(model, X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}", color="darkorange", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_roc_curve(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> plt.Figure:
    """Plot Receiver Operating Characteristic (ROC) curve."""
    y_prob = get_probabilities(model, X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="navy", lw=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    return fig


def threshold_analysis(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
) -> pd.DataFrame:
    """Evaluate precision, recall, and F1 score across multiple prediction thresholds."""
    y_prob = get_probabilities(model, X_test)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        })
    return pd.DataFrame(rows)


def feature_importance(model: Any, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
    """Extract top N important features from tree-based or linear models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        logger.warning(f"Model {type(model).__name__} does not expose feature importances or coefficients.")
        return pd.DataFrame(columns=["feature", "importance"])

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def plot_feature_importance(fi_df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """Plot horizontal bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(8, 6))
    if fi_df.empty:
        ax.text(0.5, 0.5, "No feature importances available", ha="center", va="center")
        return fig
    data = fi_df.head(top_n).iloc[::-1]
    ax.barh(data["feature"], data["importance"], color="teal")
    ax.set_title(f"Top {top_n} Important Features")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig
