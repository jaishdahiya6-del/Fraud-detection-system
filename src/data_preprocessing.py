"""Data loading, cleaning and preprocessing for fraud detection."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COL = "Class"


def load_data(path="data/creditcard.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna()
    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COL):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def stratified_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def scale_features(X_train, X_test, cols_to_scale=("Amount", "Time")):
    """Fit scaler on train only, then transform both. Avoids data leakage."""
    scaler = StandardScaler()
    cols = [c for c in cols_to_scale if c in X_train.columns]
    X_train = X_train.copy()
    X_test = X_test.copy()
    if cols:
        X_train[cols] = scaler.fit_transform(X_train[cols])
        X_test[cols] = scaler.transform(X_test[cols])
    return X_train, X_test, scaler


def preprocess_pipeline(path="data/creditcard.csv", target_col=TARGET_COL):
    df = load_data(path)
    df = clean_data(df)
    X, y = split_features_target(df, target_col)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    X_train, X_test, scaler = scale_features(X_train, X_test)
    return X_train, X_test, y_train, y_test, scaler
