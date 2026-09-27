"""Unit tests for src.predict module."""
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.predict import risk_level, predict_transaction


@pytest.fixture
def trained_model_and_scaler():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame(np.random.randn(n, 4), columns=["V1", "V2", "Amount", "Time"])
    y = pd.Series(np.random.choice([0, 1], size=n, p=[0.8, 0.2]))

    scaler = StandardScaler()
    X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model, scaler


def test_risk_level():
    assert risk_level(0.10) == "Low Risk"
    assert risk_level(0.50) == "Medium Risk"
    assert risk_level(0.85) == "High Risk"


def test_predict_transaction(trained_model_and_scaler):
    model, scaler = trained_model_and_scaler
    tx = {"V1": 0.5, "V2": -1.2, "Amount": 100.0, "Time": 500.0}
    res = predict_transaction(model, scaler, tx, threshold=0.5)

    assert "prediction" in res
    assert res["prediction"] in ["Fraud", "Genuine"]
    assert "fraud_probability" in res
    assert 0.0 <= res["fraud_probability"] <= 100.0
    assert "risk_level" in res


def test_predict_transaction_empty(trained_model_and_scaler):
    model, scaler = trained_model_and_scaler
    with pytest.raises(ValueError, match="Transaction dictionary cannot be empty"):
        predict_transaction(model, scaler, {})
