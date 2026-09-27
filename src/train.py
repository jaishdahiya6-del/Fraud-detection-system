"""Train multiple classification models with various class imbalance handling strategies."""
import os
import logging
from typing import Tuple, Dict, Any, List, Optional
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

from data_preprocessing import preprocess_pipeline
from evaluate import evaluate_model

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

RANDOM_STATE = 42

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=50, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=150, max_depth=6, tree_method="hist",
        random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1
    ),
}

RESAMPLERS = {
    "baseline": None,
    "undersample": RandomUnderSampler(random_state=RANDOM_STATE),
    "oversample": RandomOverSampler(random_state=RANDOM_STATE),
    "smote": SMOTE(random_state=RANDOM_STATE),
    "smote_tomek": SMOTETomek(random_state=RANDOM_STATE),
}

ACTIVE_RESAMPLERS = ["baseline", "undersample", "oversample", "smote"]


def get_class_weighted_models() -> Dict[str, Any]:
    """Return dict of model instances configured with class weighting."""
    return {
        "Logistic Regression (weighted)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree (weighted)": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest (weighted)": RandomForestClassifier(
            n_estimators=50, max_depth=10, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def resample(X_train: pd.DataFrame, y_train: pd.Series, method: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply specified resampling strategy to training data."""
    sampler = RESAMPLERS.get(method)
    if sampler is None:
        return X_train, y_train
    logger.info(f"Applying resampling strategy: {method}")
    return sampler.fit_resample(X_train, y_train)


def run_experiments(
    data_path: str = "data/creditcard.csv", verbose: bool = True
) -> Tuple[pd.DataFrame, Any, str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any]]:
    """Train baseline + imbalance-handled models, return comparison table + best model."""
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(data_path)

    results = []
    fitted = {}

    for resample_method in ACTIVE_RESAMPLERS:
        Xr, yr = resample(X_train, y_train, resample_method)
        for name, model in MODELS.items():
            key = f"{name} [{resample_method}]"
            if verbose:
                logger.info(f"Training {key} ...")
            model.fit(Xr, yr)
            metrics = evaluate_model(model, X_test, y_test)
            metrics["model"] = key
            results.append(metrics)
            fitted[key] = model

    # class-weighted models trained on original (unresampled) data
    for name, model in get_class_weighted_models().items():
        if verbose:
            logger.info(f"Training {name} ...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        metrics["model"] = name
        results.append(metrics)
        fitted[name] = model

    comparison_df = pd.DataFrame(results).set_index("model")
    comparison_df = comparison_df.sort_values("f1", ascending=False)

    best_name = comparison_df.index[0]
    best_model = fitted[best_name]

    return comparison_df, best_model, best_name, (X_train, X_test, y_train, y_test, scaler)


def save_model(model: Any, scaler: Any, path: str = "models/best_model.pkl") -> None:
    """Save model and scaler bundle to path, creating directories as needed."""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    logger.info(f"Saving model bundle to {path}...")
    joblib.dump({"model": model, "scaler": scaler}, path)


if __name__ == "__main__":
    comparison_df, best_model, best_name, splits = run_experiments()
    logger.info("\n" + str(comparison_df))
    logger.info(f"\nBest model: {best_name}")
    _, _, _, _, scaler = splits
    save_model(best_model, scaler)

    os.makedirs("models", exist_ok=True)
    comparison_df.to_csv("models/model_comparison.csv")
