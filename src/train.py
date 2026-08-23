"""Train multiple models with different imbalance-handling strategies."""
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

RANDOM_STATE = 42

# max_depth caps and modest tree counts keep runtime reasonable on 250k+ row
# datasets (SMOTE/oversample can double the training set size) while still
# giving representative, non-hardcoded performance.
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

# SMOTE + Tomek Links is kept available (see resample()) but excluded from
# the default grid below: Tomek-link removal needs nearest-neighbour search
# over the whole resampled set, which is very slow on 250k+ row datasets.
# Plain SMOTE already covers the "synthetic minority oversampling" objective.
RESAMPLERS = {
    "baseline": None,
    "undersample": RandomUnderSampler(random_state=RANDOM_STATE),
    "oversample": RandomOverSampler(random_state=RANDOM_STATE),
    "smote": SMOTE(random_state=RANDOM_STATE),
    "smote_tomek": SMOTETomek(random_state=RANDOM_STATE),
}

ACTIVE_RESAMPLERS = ["baseline", "undersample", "oversample", "smote"]


def get_class_weighted_models():
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


def resample(X_train, y_train, method: str):
    sampler = RESAMPLERS.get(method)
    if sampler is None:
        return X_train, y_train
    return sampler.fit_resample(X_train, y_train)


def run_experiments(data_path="data/creditcard.csv", verbose=True):
    """Train baseline + imbalance-handled models, return comparison table + best model."""
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(data_path)

    results = []
    fitted = {}

    for resample_method in ACTIVE_RESAMPLERS:
        Xr, yr = resample(X_train, y_train, resample_method)
        for name, model in MODELS.items():
            key = f"{name} [{resample_method}]"
            if verbose:
                print(f"Training {key} ...")
            model.fit(Xr, yr)
            metrics = evaluate_model(model, X_test, y_test)
            metrics["model"] = key
            results.append(metrics)
            fitted[key] = model

    # class-weighted models trained on original (unresampled) data
    for name, model in get_class_weighted_models().items():
        if verbose:
            print(f"Training {name} ...")
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


def save_model(model, scaler, path="models/best_model.pkl"):
    joblib.dump({"model": model, "scaler": scaler}, path)


if __name__ == "__main__":
    comparison_df, best_model, best_name, splits = run_experiments()
    print(comparison_df)
    print(f"\nBest model: {best_name}")
    _, _, _, _, scaler = splits
    save_model(best_model, scaler)
    comparison_df.to_csv("models/model_comparison.csv")
