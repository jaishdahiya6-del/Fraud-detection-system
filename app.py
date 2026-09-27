"""Fraud Detection Dashboard - Streamlit application."""
import sys
import os
from typing import Dict, Any, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from data_preprocessing import load_data, clean_data, split_features_target, stratified_split
from evaluate import (
    get_confusion_matrix, plot_confusion_matrix, plot_pr_curve, plot_roc_curve,
    threshold_analysis, feature_importance, plot_feature_importance, evaluate_model,
)
from predict import risk_level, predict_transaction, load_model

st.set_page_config(
    page_title="Fraud Detection & Risk Analytics System",
    layout="wide",
    page_icon="💳",
    initial_sidebar_state="expanded",
)

DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "models/best_model.pkl"
COMPARISON_PATH = "models/model_comparison.csv"


@st.cache_data
def get_data() -> Optional[pd.DataFrame]:
    if not os.path.exists(DATA_PATH):
        return None
    try:
        return clean_data(load_data(DATA_PATH))
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_resource
def get_model_bundle() -> Optional[Dict[str, Any]]:
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model, scaler = load_model(MODEL_PATH)
        return {"model": model, "scaler": scaler}
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


df = get_data()
bundle = get_model_bundle()

# Sidebar Navigation
st.sidebar.title("💳 Fraud Detection")
st.sidebar.caption("Enterprise Risk Engine & ML Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Fraud Analytics",
        "Model Performance",
        "Single Transaction Scoring",
        "Model Explainability",
    ],
)

# Status Indicator in Sidebar
st.sidebar.divider()
if df is not None:
    st.sidebar.success(f"Data Loaded: {len(df):,} rows")
else:
    st.sidebar.warning("Data File Missing")

if bundle is not None:
    model_name = type(bundle["model"]).__name__
    st.sidebar.success(f"Model Active: {model_name}")
else:
    st.sidebar.warning("Model File Missing")


if df is None:
    st.warning(
        "⚠️ No dataset found at `data/creditcard.csv`.\n\n"
        "To enable full dashboard features:\n"
        "1. Download the Kaggle Credit Card Fraud Detection dataset.\n"
        "2. Place `creditcard.csv` inside the `data/` folder.\n"
        "3. Run `python src/train.py` to train and save the model."
    )
    st.stop()

target_col = "Class"
fraud_count = int(df[target_col].sum())
genuine_count = int(len(df) - fraud_count)
fraud_rate = (fraud_count / len(df) * 100) if len(df) > 0 else 0.0

# ---------------- 1. Dashboard Overview ----------------
if page == "Dashboard Overview":
    st.title("🛡️ Fraud Detection & Risk Analytics Engine")
    st.markdown(
        "Real-time monitoring dashboard for transaction risk scoring, class imbalance evaluation, "
        "and Machine Learning model oversight."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Fraud Transactions", f"{fraud_count:,}", delta=f"{fraud_rate:.3f}% rate", delta_color="inverse")
    c3.metric("Genuine Transactions", f"{genuine_count:,}")
    c4.metric("Class Imbalance Ratio", f"1 : {int(genuine_count / (fraud_count or 1)):,}")

    st.divider()

    st.subheader("🎛️ Interactive Transaction Filtering")
    f1, f2, f3 = st.columns(3)

    amt_min, amt_max = float(df["Amount"].min()), float(df["Amount"].max())
    amount_range = f1.slider("Filter Amount Range ($)", 0.0, round(amt_max, 2), (0.0, round(amt_max, 2)))

    class_filter = f2.radio("Filter Class", ["All", "Fraud Only", "Genuine Only"], horizontal=True)

    if "Time" in df.columns:
        t_min, t_max = int(df["Time"].min()), int(df["Time"].max())
        time_range = f3.slider("Filter Time Elapsed (s)", t_min, t_max, (t_min, t_max))
    else:
        time_range = None

    filtered = df[(df["Amount"] >= amount_range[0]) & (df["Amount"] <= amount_range[1])]
    if time_range and "Time" in df.columns:
        filtered = filtered[(filtered["Time"] >= time_range[0]) & (filtered["Time"] <= time_range[1])]
    if class_filter == "Fraud Only":
        filtered = filtered[filtered[target_col] == 1]
    elif class_filter == "Genuine Only":
        filtered = filtered[filtered[target_col] == 0]

    fc1, fc2, fc3 = st.columns(3)
    f_total = len(filtered)
    f_fraud = int((filtered[target_col] == 1).sum()) if not filtered.empty else 0
    fc1.metric("Filtered Volume", f"{f_total:,}")
    fc2.metric("Filtered Frauds", f"{f_fraud:,}")
    fc3.metric("Filtered Fraud Rate", f"{(f_fraud / f_total * 100) if f_total else 0:.4f}%")

    if filtered.empty:
        st.warning("No transactions match current filters.")
    else:
        a1, a2 = st.columns(2)
        with a1:
            pie_counts = filtered[target_col].value_counts().rename({0: "Genuine", 1: "Fraud"})
            fig = px.pie(
                names=pie_counts.index,
                values=pie_counts.values,
                title="Class Breakdown",
                color=pie_counts.index,
                color_discrete_map={"Genuine": "#2E86AB", "Fraud": "#E63946"},
            )
            st.plotly_chart(fig, use_container_width=True)
        with a2:
            fig2 = px.histogram(
                filtered,
                x="Amount",
                color=target_col,
                nbins=50,
                barmode="overlay",
                title="Amount Distribution by Class",
                color_discrete_map={0: "#2E86AB", 1: "#E63946"},
            )
            st.plotly_chart(fig2, use_container_width=True)

# ---------------- 2. Fraud Analytics ----------------
elif page == "Fraud Analytics":
    st.title("📊 Deep-Dive Fraud Analytics")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            names=["Genuine", "Fraud"],
            values=[genuine_count, fraud_count],
            title="Overall Class Distribution",
            color_discrete_sequence=["#2E86AB", "#E63946"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.histogram(
            df,
            x="Amount",
            color=target_col,
            nbins=50,
            barmode="overlay",
            title="Transaction Amount Distribution",
            color_discrete_map={0: "#2E86AB", 1: "#E63946"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    if "Time" in df.columns:
        df_time = df.copy()
        df_time["Hour"] = (df_time["Time"] // 3600) % 24
        hourly = df_time.groupby(["Hour", target_col]).size().reset_index(name="Count")
        fig_time = px.line(
            hourly,
            x="Hour",
            y="Count",
            color=target_col,
            title="Hourly Transaction Pattern (24-Hour Cycle)",
            color_discrete_map={0: "#2E86AB", 1: "#E63946"},
        )
        st.plotly_chart(fig_time, use_container_width=True)

# ---------------- 3. Model Performance ----------------
elif page == "Model Performance":
    st.title("📈 Model Performance & Evaluation Metrics")

    if os.path.exists(COMPARISON_PATH):
        st.subheader("Model & Resampling Strategy Comparison Grid")
        comp_df = pd.read_csv(COMPARISON_PATH, index_col=0)
        st.dataframe(comp_df.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)

        fig_comp = px.bar(
            comp_df.reset_index(),
            x="model",
            y="f1",
            color="f1",
            title="F1-Score Comparison Across Experiments",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Run `python src/train.py` to generate the benchmark comparison matrix.")

    if bundle:
        st.divider()
        st.subheader("Active Model Diagnostic Curves")
        model, scaler = bundle["model"], bundle["scaler"]
        X, y = split_features_target(df, target_col)
        X_train, X_test, y_train, y_test = stratified_split(X, y)
        cols = [c for c in ["Amount", "Time"] if c in X_test.columns]
        X_test_scaled = X_test.copy()
        if cols:
            X_test_scaled[cols] = scaler.transform(X_test[cols])

        c1, c2 = st.columns(2)
        with c1:
            cm = get_confusion_matrix(model, X_test_scaled, y_test)
            st.pyplot(plot_confusion_matrix(cm["matrix"]))
            st.caption(
                f"**TP**: {cm['tp']} (Caught Fraud) | **FN**: {cm['fn']} (Missed Fraud)\n\n"
                f"**FP**: {cm['fp']} (False Alarm) | **TN**: {cm['tn']} (Genuine)"
            )
        with c2:
            st.pyplot(plot_pr_curve(model, X_test_scaled, y_test))

        st.pyplot(plot_roc_curve(model, X_test_scaled, y_test))

        st.subheader("Decision Threshold Optimization Matrix")
        st.caption("Adjust decision thresholds to prioritize Recall (reducing missed fraud) vs Precision.")
        t_df = threshold_analysis(model, X_test_scaled, y_test)
        st.dataframe(t_df, use_container_width=True)

# ---------------- 4. Single Transaction Scoring ----------------
elif page == "Single Transaction Scoring":
    st.title("🎯 Single Transaction Scoring & Inference")

    if not bundle:
        st.warning("No active model bundle found. Train a model using `python src/train.py` first.")
    else:
        model, scaler = bundle["model"], bundle["scaler"]
        feature_cols = [c for c in df.columns if c != target_col]

        threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.05)

        st.subheader("Input Feature Values")
        input_vals = {}
        n_cols = 4
        cols = st.columns(n_cols)
        for i, feat in enumerate(feature_cols):
            default = float(df[feat].median())
            input_vals[feat] = cols[i % n_cols].number_input(feat, value=default, key=f"inp_{feat}")

        if st.button("🔍 Evaluate Transaction Risk", type="primary"):
            res = predict_transaction(model, scaler, input_vals, threshold=threshold)

            m1, m2, m3 = st.columns(3)
            m1.metric("Prediction", res["prediction"])
            m2.metric("Fraud Probability", f"{res['fraud_probability']:.2f}%")
            m3.metric("Risk Categorization", res["risk_level"])

            if res["prediction"] == "Fraud":
                st.error("🚨 High-risk transaction detected! Recommended Action: Flag for manual review.")
            else:
                st.success("✅ Low-risk transaction detected. Action: Pass.")

# ---------------- 5. Model Explainability ----------------
elif page == "Model Explainability":
    st.title("💡 Model Explainability & Feature Importances")

    if not bundle:
        st.warning("No active model bundle found. Train a model using `python src/train.py` first.")
    else:
        model = bundle["model"]
        feature_cols = [c for c in df.columns if c != target_col]
        fi = feature_importance(model, feature_cols, top_n=20)

        if fi.empty:
            st.info("The active model does not support direct feature importance extraction.")
        else:
            st.pyplot(plot_feature_importance(fi, top_n=15))
            st.dataframe(fi, use_container_width=True)
