"""Load saved model and predict fraud probability for new transactions."""
import joblib
import pandas as pd


def load_model(path="models/best_model.pkl"):
    bundle = joblib.load(path)
    return bundle["model"], bundle["scaler"]


def risk_level(prob: float) -> str:
    if prob < 0.30:
        return "Low Risk"
    elif prob < 0.70:
        return "Medium Risk"
    return "High Risk"


def predict_transaction(model, scaler, transaction: dict, threshold: float = 0.5,
                         scale_cols=("Amount", "Time")):
    """transaction: dict of feature_name -> value (must match training columns)."""
    X = pd.DataFrame([transaction])
    cols = [c for c in scale_cols if c in X.columns]
    if cols:
        X[cols] = scaler.transform(X[cols])

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1][0]
    else:
        prob = model.decision_function(X)[0]

    prediction = "Fraud" if prob >= threshold else "Genuine"
    return {
        "prediction": prediction,
        "fraud_probability": round(float(prob) * 100, 2),
        "risk_level": risk_level(prob),
    }
