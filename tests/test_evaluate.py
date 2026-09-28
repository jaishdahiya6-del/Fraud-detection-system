"""Unit tests for src.evaluate module."""
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.evaluate import (
    evaluate_model,
    get_confusion_matrix,
    threshold_analysis,
    feature_importance,
)


@pytest.fixture
def trained_model_data():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame(np.random.randn(n, 4), columns=["V1", "V2", "Amount", "Time"])
    y = pd.Series(np.random.choice([0, 1], size=n, p=[0.8, 0.2]))
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model, X, y


def test_evaluate_model(trained_model_data):
    model, X, y = trained_model_data
    metrics = evaluate_model(model, X, y)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert 0.0 <= metrics["f1"] <= 1.0


def test_get_confusion_matrix(trained_model_data):
    model, X, y = trained_model_data
    cm_dict = get_confusion_matrix(model, X, y)
    assert "tn" in cm_dict
    assert "fp" in cm_dict
    assert "fn" in cm_dict
    assert "tp" in cm_dict
    assert cm_dict["tn"] + cm_dict["fp"] + cm_dict["fn"] + cm_dict["tp"] == len(y)


def test_threshold_analysis(trained_model_data):
    model, X, y = trained_model_data
    thresh_df = threshold_analysis(model, X, y, thresholds=[0.3, 0.5, 0.7])
    assert len(thresh_df) == 3
    assert list(thresh_df.columns) == ["threshold", "precision", "recall", "f1"]


def test_feature_importance(trained_model_data):
    model, X, y = trained_model_data
    fi_df = feature_importance(model, list(X.columns), top_n=2)
    assert len(fi_df) <= 2
    assert "feature" in fi_df.columns
    assert "importance" in fi_df.columns
