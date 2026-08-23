"""Fraud Detection Dashboard - Streamlit app."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from data_preprocessing import load_data, clean_data
from evaluate import (
    get_confusion_matrix, plot_confusion_matrix, plot_pr_curve, plot_roc_curve,
    threshold_analysis, feature_importance, plot_feature_importance, evaluate_model,
)
from predict import risk_level

st.set_page_config(page_title="Fraud Detection System", layout="wide", page_icon="💳")

DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "models/best_model.pkl"
COMPARISON_PATH = "models/model_comparison.csv"


@st.cache_data
def get_data():
    if not os.path.exists(DATA_PATH):
        return None
    return clean_data(load_data(DATA_PATH))


@st.cache_resource
def get_model_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


df = get_data()
bundle = get_model_bundle()

st.sidebar.title("💳 Fraud Detection")
page = st.sidebar.radio("Navigate", [
    "Dashboard", "Fraud Analytics", "Model Comparison",
    "Transaction Prediction", "Model Explainability",
])

if df is None:
    st.warning(
        "No dataset found. Place `creditcard.csv` inside the `data/` folder "
        "(Kaggle Credit Card Fraud Detection dataset) and run `python src/train.py` "
        "to train the model before using this dashboard."
    )
    st.stop()

target_col = "Class"
fraud_count = int(df[target_col].sum())
genuine_count = int(len(df) - fraud_count)
fraud_rate = round(fraud_count / len(df) * 100, 4)

# ---------------- Dashboard ----------------
if page == "Dashboard":
    st.title("Fraud Detection & Risk Analytics System")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Fraud Transactions", f"{fraud_count:,}")
    c3.metric("Genuine Transactions", f"{genuine_count:,}")
    c4.metric("Fraud Rate", f"{fraud_rate}%")

    if bundle:
        model, scaler = bundle["model"], bundle["scaler"]
        from data_preprocessing import split_features_target, stratified_split
        X, y = split_features_target(df, target_col)
        X_train, X_test, y_train, y_test = stratified_split(X, y)
        cols = [c for c in ["Amount", "Time"] if c in X_test.columns]
        X_test_scaled = X_test.copy()
        if cols:
            X_test_scaled[cols] = scaler.transform(X_test[cols])
        m = evaluate_model(model, X_test_scaled, y_test)
        c1, c2, c3 = st.columns(3)
        c1.metric("Model F1 Score", f"{m['f1']:.3f}")
        c2.metric("Model Recall", f"{m['recall']:.3f}")
        c3.metric("Model Precision", f"{m['precision']:.3f}")
    else:
        st.info("Train a model with `python src/train.py` to see model metrics here.")

    st.divider()

    # ---------- Interactive filters: let the user reshape what the charts show ----------
    st.subheader("🎛️ Filters — adjust to explore the data yourself")
    f1, f2, f3 = st.columns(3)

    amt_min, amt_max = float(df["Amount"].min()), float(df["Amount"].max())
    amount_range = f1.slider(
        "Amount range", min_value=0.0, max_value=round(amt_max, 2),
        value=(0.0, round(amt_max, 2)),
    )

    class_filter = f2.radio("Show", ["All", "Fraud only", "Genuine only"], horizontal=True)

    if "Time" in df.columns:
        t_min, t_max = int(df["Time"].min()), int(df["Time"].max())
        time_range = f3.slider("Time range (seconds elapsed)", t_min, t_max, (t_min, t_max))
    else:
        time_range = None

    filtered = df[(df["Amount"] >= amount_range[0]) & (df["Amount"] <= amount_range[1])]
    if time_range and "Time" in df.columns:
        filtered = filtered[(filtered["Time"] >= time_range[0]) & (filtered["Time"] <= time_range[1])]
    if class_filter == "Fraud only":
        filtered = filtered[filtered[target_col] == 1]
    elif class_filter == "Genuine only":
        filtered = filtered[filtered[target_col] == 0]

    fc1, fc2, fc3 = st.columns(3)
    f_fraud = int((filtered[target_col] == 1).sum())
    f_total = len(filtered)
    fc1.metric("Filtered Transactions", f"{f_total:,}")
    fc2.metric("Filtered Fraud Count", f"{f_fraud:,}")
    fc3.metric("Filtered Fraud Rate", f"{(f_fraud / f_total * 100) if f_total else 0:.4f}%")

    if filtered.empty:
        st.warning("No transactions match the current filters — widen the ranges above.")
    else:
        st.subheader("📊 Analysis")
        a1, a2 = st.columns(2)
        with a1:
            pie_counts = filtered[target_col].value_counts().rename({0: "Genuine", 1: "Fraud"})
            fig = px.pie(names=pie_counts.index, values=pie_counts.values,
                         title="Fraud vs Genuine (filtered)", color=pie_counts.index,
                         color_discrete_map={"Genuine": "#2E86AB", "Fraud": "#E63946"})
            st.plotly_chart(fig, use_container_width=True)
        with a2:
            fig2 = px.histogram(filtered, x="Amount", color=target_col, nbins=50, barmode="overlay",
                                 title="Amount Distribution (filtered)",
                                 color_discrete_map={0: "#2E86AB", 1: "#E63946"})
            st.plotly_chart(fig2, use_container_width=True)

        if "Time" in filtered.columns and not filtered.empty:
            trend = filtered.copy()
            trend["hour"] = (trend["Time"] // 3600).astype(int)
            trend_agg = trend.groupby(["hour", target_col]).size().reset_index(name="count")
            fig3 = px.line(trend_agg, x="hour", y="count", color=target_col,
                            title="Transaction Volume Over Time (by hour)",
                            color_discrete_map={0: "#2E86AB", 1: "#E63946"})
            st.plotly_chart(fig3, use_container_width=True)

        # Correlation of numeric features with Class, on filtered data
        num_df = filtered.select_dtypes(include=[np.number])
        if target_col in num_df.columns and num_df.shape[1] > 1:
            corr_with_target = num_df.corr(numeric_only=True)[target_col].drop(target_col)
            corr_with_target = corr_with_target.reindex(
                corr_with_target.abs().sort_values(ascending=False).index
            ).head(15)
            fig4 = px.bar(
                x=corr_with_target.values, y=corr_with_target.index, orientation="h",
                title="Top Features Correlated with Fraud (filtered)",
                labels={"x": "Correlation with Class", "y": "Feature"},
            )
            st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ---------- Self-service chart builder ----------
    st.subheader("🛠️ Build Your Own Chart")
    st.caption("Pick a chart type and columns — the chart updates live from the filtered data above.")
    numeric_cols = [c for c in df.columns if c != target_col]

    b1, b2, b3, b4 = st.columns(4)
    chart_type = b1.selectbox(
        "Chart type", ["Histogram", "Scatter", "Box", "Bar (avg by class)", "Line", "Violin"]
    )
    x_col = b2.selectbox("X-axis column", numeric_cols, index=numeric_cols.index("Amount") if "Amount" in numeric_cols else 0)
    y_col = b3.selectbox("Y-axis column (if applicable)", ["(none)"] + numeric_cols)
    color_by_class = b4.checkbox("Color by Class", value=True)

    sample_n = st.slider("Sample size (for performance on large data)", 500, min(50000, len(filtered) or 500),
                          min(5000, len(filtered) or 500))
    plot_data = filtered.sample(min(sample_n, len(filtered)), random_state=42) if not filtered.empty else filtered
    color_arg = target_col if color_by_class else None

    try:
        if plot_data.empty:
            st.info("No data available for the current filters to build a chart.")
        elif chart_type == "Histogram":
            fig_c = px.histogram(plot_data, x=x_col, color=color_arg, nbins=50, barmode="overlay",
                                  title=f"Histogram of {x_col}")
            st.plotly_chart(fig_c, use_container_width=True)
        elif chart_type == "Scatter":
            y_use = y_col if y_col != "(none)" else x_col
            fig_c = px.scatter(plot_data, x=x_col, y=y_use, color=color_arg,
                                title=f"{x_col} vs {y_use}", opacity=0.6)
            st.plotly_chart(fig_c, use_container_width=True)
        elif chart_type == "Box":
            fig_c = px.box(plot_data, x=target_col if color_by_class else None, y=x_col,
                            title=f"Box Plot of {x_col}")
            st.plotly_chart(fig_c, use_container_width=True)
        elif chart_type == "Bar (avg by class)":
            bar_data = plot_data.groupby(target_col)[x_col].mean().reset_index()
            fig_c = px.bar(bar_data, x=target_col, y=x_col, title=f"Average {x_col} by Class")
            st.plotly_chart(fig_c, use_container_width=True)
        elif chart_type == "Line":
            y_use = y_col if y_col != "(none)" else x_col
            line_data = plot_data.sort_values(x_col)
            fig_c = px.line(line_data, x=x_col, y=y_use, color=color_arg, title=f"{y_use} over {x_col}")
            st.plotly_chart(fig_c, use_container_width=True)
        elif chart_type == "Violin":
            fig_c = px.violin(plot_data, x=target_col if color_by_class else None, y=x_col,
                               box=True, title=f"Violin Plot of {x_col}")
            st.plotly_chart(fig_c, use_container_width=True)
    except Exception as e:
        st.error(f"Couldn't build that chart with the selected columns: {e}")

# ---------------- Fraud Analytics ----------------
elif page == "Fraud Analytics":
    st.title("Fraud Analytics")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(names=["Genuine", "Fraud"], values=[genuine_count, fraud_count],
                     title="Fraud vs Genuine")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.histogram(df, x="Amount", color=target_col, nbins=50, barmode="overlay",
                             title="Transaction Amount Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    if bundle:
        model, scaler = bundle["model"], bundle["scaler"]
        from data_preprocessing import split_features_target, stratified_split
        X, y = split_features_target(df, target_col)
        X_train, X_test, y_train, y_test = stratified_split(X, y)
        cols = [c for c in ["Amount", "Time"] if c in X_test.columns]
        X_test_scaled = X_test.copy()
        if cols:
            X_test_scaled[cols] = scaler.transform(X_test[cols])

        col3, col4 = st.columns(2)
        with col3:
            cm = get_confusion_matrix(model, X_test_scaled, y_test)
            st.pyplot(plot_confusion_matrix(cm["matrix"]))
            st.caption(
                f"TP={cm['tp']}  FN={cm['fn']} (missed fraud)  "
                f"FP={cm['fp']} (genuine flagged)  TN={cm['tn']}"
            )
        with col4:
            st.pyplot(plot_pr_curve(model, X_test_scaled, y_test))
        st.pyplot(plot_roc_curve(model, X_test_scaled, y_test))

        st.subheader("Threshold Optimization")
        t_df = threshold_analysis(model, X_test_scaled, y_test)
        st.dataframe(t_df, use_container_width=True)
        fig3 = px.line(t_df, x="threshold", y=["precision", "recall", "f1"],
                        title="Precision / Recall / F1 vs Threshold")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Train a model to see confusion matrix, PR/ROC curves and threshold analysis.")

# ---------------- Model Comparison ----------------
elif page == "Model Comparison":
    st.title("Model Comparison")
    if os.path.exists(COMPARISON_PATH):
        comp_df = pd.read_csv(COMPARISON_PATH, index_col=0)
        st.dataframe(comp_df.style.highlight_max(axis=0, color="lightgreen"),
                     use_container_width=True)
        fig = px.bar(comp_df.reset_index(), x="model", y="f1", title="F1 Score by Model")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `python src/train.py` to generate the model comparison table.")

# ---------------- Transaction Prediction ----------------
elif page == "Transaction Prediction":
    st.title("Transaction Prediction")
    if not bundle:
        st.info("Train a model first with `python src/train.py`.")
    else:
        model, scaler = bundle["model"], bundle["scaler"]
        feature_cols = [c for c in df.columns if c != target_col]
        threshold = st.sidebar.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)

        st.write("Enter transaction feature values:")
        input_vals = {}
        n_cols = 4
        cols = st.columns(n_cols)
        for i, feat in enumerate(feature_cols):
            default = float(df[feat].median())
            input_vals[feat] = cols[i % n_cols].number_input(feat, value=default)

        if st.button("Detect Fraud"):
            X = pd.DataFrame([input_vals])
            scale_cols = [c for c in ["Amount", "Time"] if c in X.columns]
            if scale_cols:
                X[scale_cols] = scaler.transform(X[scale_cols])
            prob = model.predict_proba(X)[:, 1][0] if hasattr(model, "predict_proba") \
                else model.decision_function(X)[0]
            pred = "Fraud" if prob >= threshold else "Genuine"
            level = risk_level(prob)

            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", pred)
            c2.metric("Fraud Probability", f"{prob*100:.2f}%")
            c3.metric("Risk Level", level)

            fi = feature_importance(model, feature_cols, top_n=10)
            if not fi.empty:
                st.subheader("Top Contributing Features")
                st.pyplot(plot_feature_importance(fi, top_n=10))

# ---------------- Model Explainability ----------------
elif page == "Model Explainability":
    st.title("Model Explainability")
    if not bundle:
        st.info("Train a model first with `python src/train.py`.")
    else:
        model = bundle["model"]
        feature_cols = [c for c in df.columns if c != target_col]
        fi = feature_importance(model, feature_cols, top_n=20)
        if fi.empty:
            st.info("Selected model does not expose feature importances directly.")
        else:
            st.pyplot(plot_feature_importance(fi, top_n=20))
            st.dataframe(fi, use_container_width=True)

        try:
            import shap
            st.subheader("SHAP Summary (sample)")
            from data_preprocessing import split_features_target, stratified_split
            X, y = split_features_target(df, target_col)
            X_train, X_test, y_train, y_test = stratified_split(X, y)
            sample = X_test.sample(min(200, len(X_test)), random_state=42)
            explainer = shap.Explainer(model, sample)
            shap_values = explainer(sample)
            import matplotlib.pyplot as plt
            fig = plt.figure()
            shap.summary_plot(shap_values, sample, show=False)
            st.pyplot(fig)
        except Exception:
            st.caption("SHAP explanations unavailable for this model/setup (optional feature).")
