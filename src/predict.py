"""Inference and risk scoring functions for new transaction predictions."""
import os
import logging
from typing import Dict, Tuple, Any, Optional
import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def load_model(path: str = "models/best_model.pkl") -> Tuple[Any, Any]:
    """Load saved model and scaler bundle from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at path: {path}")
    logger.info(f"Loading model bundle from {path}...")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle or "scaler" not in bundle:
        raise ValueError("Loaded model bundle must be a dict containing 'model' and 'scaler' keys.")
    return bundle["model"], bundle["scaler"]


def risk_level(prob: float) -> str:
    """Categorize fraud risk probability into Low, Medium, or High Risk."""
    if prob < 0.30:
        return "Low Risk"
    elif prob < 0.70:
        return "Medium Risk"
    return "High Risk"


def predict_transaction(
    model: Any,
    scaler: Any,
    transaction: Dict[str, float],
    threshold: float = 0.5,
    scale_cols: Tuple[str, ...] = ("Amount", "Time")
) -> Dict[str, Any]:
    """Predict fraud probability and risk level for a single transaction dictionary.

    Args:
        model: Trained classifier model.
        scaler: Fitted scaler for features.
        transaction: Dictionary mapping feature names to numerical values.
        threshold: Decision threshold for fraud classification.
        scale_cols: Tuple of column names to scale.

    Returns:
        Dict containing prediction, fraud_probability percentage, and risk_level.
    """
    if not transaction:
        raise ValueError("Transaction dictionary cannot be empty.")

    X = pd.DataFrame([transaction])
    cols = [c for c in scale_cols if c in X.columns]
    if cols:
        X[cols] = scaler.transform(X[cols])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] > 1:
            prob = float(probs[:, 1][0])
        else:
            prob = float(probs.ravel()[0])
    elif hasattr(model, "decision_function"):
        prob = float(model.decision_function(X)[0])
    else:
        raise AttributeError("Model does not support `predict_proba` or `decision_function`.")

    prediction = "Fraud" if prob >= threshold else "Genuine"
    return {
        "prediction": prediction,
        "fraud_probability": round(prob * 100, 2),
        "risk_level": risk_level(prob),
    }
