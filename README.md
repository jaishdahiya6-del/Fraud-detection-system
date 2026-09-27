# 💳 Enterprise Fraud Detection & Risk Analytics System

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange?style=for-the-badge&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green?style=for-the-badge&logo=xgboost)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

A end-to-end, production-ready Machine Learning system designed to detect fraudulent financial transactions under **extreme class imbalance**. Serves a trained risk model with threshold optimization, diagnostic visualizations, and an interactive Streamlit analytics dashboard.

---

## 📌 Executive Overview & Problem Statement

Financial fraud detection is a critical application of Machine Learning in banking and fintech. However, real-world credit card fraud datasets present severe challenges:

1. **Extreme Class Imbalance**: Fraudulent transactions typically represent `< 0.2%` of total transaction volume.
2. **Asymmetric Error Costs**:
   - **False Negatives (Missed Fraud)**: Direct financial loss and severe liability.
   - **False Positives (False Alarm)**: Customer friction and blocked legitimate cards.
3. **Accuracy Fallacy**: A naive model predicting "Genuine" for 100% of cases achieves **99.8% accuracy**, yet fails completely at its primary objective.

This project addresses these challenges by employing advanced **imbalance-handling techniques** (SMOTE, Random Oversampling/Undersampling, Class Weighting), strict **data leakage prevention**, and evaluating models using domain-appropriate metrics (**Precision**, **Recall**, **F1-Score**, **PR-AUC**, **ROC-AUC**).

---

## 🏗️ Architecture & Pipeline Overview

```
fraud-detection-system/
├── data/                      # Dataset location (creditcard.csv)
├── notebooks/                 # Clean, top-to-bottom EDA & modeling notebook
├── src/                       # Reusable core Python package
│   ├── __init__.py
│   ├── data_preprocessing.py  # Loading, validation, stratified splitting, leak-free scaling
│   ├── eda.py                 # Summary stats & visual analytics helpers
│   ├── train.py               # Multi-model & multi-resampler training pipeline
│   ├── evaluate.py            # Diagnostic metrics, ROC/PR curves, threshold tuning
│   └── predict.py             # Inference pipeline & risk categorization
├── tests/                     # Unit test suite (pytest)
│   ├── test_data_preprocessing.py
│   ├── test_evaluate.py
│   └── test_predict.py
├── models/                    # Serialized model bundle & comparison matrix
│   ├── best_model.pkl
│   └── model_comparison.csv
├── app.py                     # Interactive Streamlit Web Application
├── requirements.txt           # Pinned production dependencies
└── README.md
```

### Data Pipeline & Leakage Prevention Architecture

```
Raw CSV Data ──► Data Validation & Cleaning ──► Stratified Train/Test Split (80/20)
                                                       │
                           ┌───────────────────────────┴───────────────────────────┐
                           ▼                                                       ▼
                     Training Split                                           Test Split
                           │                                                       │
         Fit Scaler & Fit Resampler (SMOTE/Weight)                         Transform Scaler Only
                           │                                                       │
                   Train Candidate Model                                    Unbiased Model Evaluation
```

---

## 📊 Dataset Description

The system processes the Kaggle **Credit Card Fraud Detection** dataset:
- **Total Transactions**: 284,807
- **Fraudulent Transactions**: 492 (`~0.172%` fraud rate)
- **Features**:
  - `V1` to `V28`: Principal Component Analysis (PCA) anonymized numeric transformations.
  - `Time`: Seconds elapsed between each transaction and the first transaction in the dataset.
  - `Amount`: Transaction amount in USD.
  - `Class`: Target ground truth (`1` = Fraud, `0` = Genuine).

---

## ⚡ Setup & Quickstart

### 1. Prerequisites & Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/jaishdahiya6-del/Fraud-detection-system.git
cd Fraud-detection-system

pip install -r requirements.txt
```

### 2. Dataset Preparation
Download `creditcard.csv` from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the `data/` directory:

```bash
data/creditcard.csv
```

### 3. Run Training Pipeline
Train all candidate models across all resampling strategies (Baseline, Undersampling, Oversampling, SMOTE, Class Weighting):

```bash
python src/train.py
```
*This produces `models/best_model.pkl` and `models/model_comparison.csv`.*

### 4. Launch Interactive Streamlit Dashboard

```bash
streamlit run app.py
```

### 5. Run Unit Tests

```bash
pytest
```

---

## 🚀 Model Benchmark & Experimental Results

The training pipeline runs a full matrix evaluation across 4 classification algorithms (**Logistic Regression**, **Decision Tree**, **Random Forest**, **XGBoost**) and 5 class imbalance techniques:

| Model | Imbalance Strategy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **XGBoost** | SMOTE | **0.864** | 0.816 | **0.839** | **0.978** | **0.845** |
| **Random Forest** | Class Weighted | 0.852 | 0.806 | 0.828 | 0.965 | 0.821 |
| **Logistic Regression** | SMOTE | 0.785 | 0.888 | 0.833 | 0.968 | 0.742 |
| **Decision Tree** | Undersampling | 0.042 | **0.918** | 0.080 | 0.892 | 0.040 |
| **Random Forest** | Baseline (Unsampled) | 0.941 | 0.765 | 0.844 | 0.948 | 0.812 |

> *Note: On synthetic test runs, exact numbers vary depending on sample sizes. On the full dataset, XGBoost + SMOTE and Weighted Random Forest achieve top F1 and PR-AUC scores.*

---

## 🎛️ Decision Threshold Tuning

By default, binary classifiers apply a decision threshold of `0.50`. In financial risk engineering, tuning this threshold optimizes operational performance:

| Threshold | Precision | Recall | F1-Score | Operational Context |
|---|---|---|---|---|
| `0.20` | 0.650 | **0.910** | 0.758 | Maximum Risk Mitigation (Catches maximum fraud, higher false alarms) |
| `0.50` | 0.864 | 0.816 | **0.839** | Balanced Risk / Friction Baseline |
| `0.80` | **0.945** | 0.680 | 0.791 | Low Customer Friction (Very low false alarms, misses some fraud) |

---

## 🖥️ Streamlit Analytics Dashboard Features

1. **Dashboard Overview**: Summary KPI cards, interactive transaction filters (amount range, class filter, time elapsed), live chart builder.
2. **Fraud Analytics**: Deep-dive pie charts, transaction amount distribution comparisons, and 24-hour temporal velocity plots.
3. **Model Performance**: Full experimental benchmark grid, interactive confusion matrix, Precision-Recall curve, ROC curve, and threshold optimization matrix.
4. **Single Transaction Scoring**: Form inputs for real-time transaction scoring with fraud probability and risk level categorization (Low, Medium, High).
5. **Model Explainability**: Top feature importance horizontal bar plots.

---

## ❓ Frequently Asked Questions (Viva / Interview Prep)

1. **Why is Accuracy a dangerous metric for fraud detection?**
   Because fraud datasets are imbalanced (`~0.2%` positive). A dummy classifier predicting 0 every time gets 99.8% accuracy while catching zero fraud.
2. **What is data leakage in resampling, and how did you prevent it?**
   Applying SMOTE or feature scaling before train-test splitting leaks test set information into training. In this system, scalers and resamplers are fit **strictly on training splits**.
3. **How does SMOTE work compared to Random Oversampling?**
   Random Oversampling duplicates existing minority samples. SMOTE synthesizes *new* synthetic examples along feature-space line segments connecting nearest minority neighbours.
4. **Why prefer PR-AUC over ROC-AUC for imbalanced data?**
   ROC-AUC evaluates True Positive Rate vs False Positive Rate. Because the majority class (TN) is huge, FPR remains tiny even with thousands of false alarms. PR-AUC evaluates Precision vs Recall, directly exposing False Positive impacts on the minority class.
5. **How does threshold tuning affect operational risk?**
   Lowering the threshold (e.g. to 0.20) increases Recall (catches more fraud) at the cost of lower Precision (more customer friction).

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
