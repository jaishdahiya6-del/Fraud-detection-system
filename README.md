# Fraud Detection with Imbalanced Data

## Overview
A financial transaction is checked and marked **Fraud** or **Genuine** using
machine learning. The core challenge: fraud is rare (often <1% of
transactions), so the project is built around techniques for **imbalanced
classification**, not just plugging data into a model.

## Problem Statement
Fraud detection is hard because fraudulent transactions are extremely rare
compared to genuine ones. A model can score 99% accuracy by predicting
"genuine" every time — while catching zero fraud. Real evaluation needs
metrics that expose this.

## Why Imbalanced Data Matters
When one class dominates (genuine ≈ 99.8%, fraud ≈ 0.2%), models naturally
bias toward the majority class. Left unhandled, this produces a model that
looks accurate but is practically useless for catching fraud.

## Objectives
- Load, clean and explore transaction data
- Quantify and visualize class imbalance
- Preprocess without data leakage (fit scalers/samplers on train only)
- Train and compare Logistic Regression, Decision Tree, Random Forest, XGBoost
- Apply and compare imbalance-handling techniques: class weighting,
  undersampling, oversampling, SMOTE, SMOTE+Tomek
- Evaluate with Precision, Recall, F1, ROC-AUC, PR-AUC — not accuracy alone
- Optimize the decision threshold instead of assuming 0.5
- Explain predictions via feature importance / SHAP
- Serve everything through an interactive Streamlit dashboard

## Dataset
Kaggle **Credit Card Fraud Detection** dataset
(https://www.kaggle.com/mlg-ulb/creditcardfraud): anonymized PCA features
`V1`–`V28`, plus `Time`, `Amount`, and target `Class` (1 = Fraud, 0 =
Genuine). Place the CSV at `data/creditcard.csv` — see `data/README.md`.
No dataset is bundled; all metrics in this project come from your own run,
never hardcoded.

## Technologies Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, Imbalanced-Learn,
XGBoost, Joblib, Plotly, Streamlit, SHAP (optional explainability).

## Machine Learning Models
- **Logistic Regression** — simple, interpretable linear baseline
- **Decision Tree** — captures non-linear splits, easy to explain
- **Random Forest** — ensemble of trees, reduces overfitting
- **XGBoost** — gradient-boosted trees, usually strongest on tabular data

## Imbalance Handling
- **Class weighting** — penalize misclassifying the minority (fraud) class
  more heavily during training (`class_weight="balanced"`)
- **Random undersampling** — remove majority-class samples to balance ratios
- **Random oversampling** — duplicate minority-class samples
- **SMOTE** — generates *synthetic* minority examples by interpolating
  between real fraud cases, instead of just duplicating them
- **SMOTE + Tomek Links** — SMOTE followed by removing borderline/ambiguous
  pairs, cleaning the decision boundary. The hybrid Tomek-link step is
  implemented in `src/train.py` (`resample(..., "smote_tomek")`) but left
  out of the default comparison grid because nearest-neighbour link removal
  is very slow on the full 280k-row dataset. It can be enabled by adding
  `"smote_tomek"` to `ACTIVE_RESAMPLERS` in `src/train.py` if you have time
  to spare.

All resampling is fit **only on the training split** — the test set always
reflects the real-world imbalance.

## Evaluation Metrics
- **Precision** — of transactions flagged fraud, how many really were
- **Recall** — of actual fraud transactions, how many were caught
- **F1-score** — harmonic mean of precision and recall
- **ROC-AUC** — ranking quality across all thresholds
- **PR-AUC** — precision/recall trade-off quality; more informative than
  ROC-AUC when the positive class is rare

## Project Architecture
```
fraud-detection-system/
├── data/                 # place creditcard.csv here
├── notebooks/             # exploratory notebook
├── src/
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── train.py           # trains all model/resampling combinations
│   ├── evaluate.py         # metrics, curves, threshold, importance
│   └── predict.py
├── models/                # saved best_model.pkl + comparison table
├── app.py                 # Streamlit dashboard
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt

# 1. Place data/creditcard.csv (see data/README.md)

# 2. Train all models and save the best one
python src/train.py

# 3. Launch the dashboard
streamlit run app.py
```

> On Windows, if `python` isn't recognized, use `py` instead (`py -m pip install -r requirements.txt`, `py src/train.py`).

Training runs 4 models × 4 imbalance-handling strategies (16 fits) plus 3
class-weighted variants on the full ~285k-row dataset. Expect roughly
2–10 minutes depending on your CPU — Random Forest and the oversampled/SMOTE
splits (which roughly double the training rows) are the slowest part.

## Results
*(Fill in after running `python src/train.py` on the real dataset — this
project does not ship pre-computed numbers.)*

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | | | | | |
| Decision Tree | | | | | |
| Random Forest | | | | | |
| XGBoost | | | | | |

**Best model:** _(fill in)_
**Chosen threshold:** _(fill in, with reasoning)_
**False negatives / false positives at chosen threshold:** _(fill in)_

## Future Improvements
- Real-time transaction monitoring / streaming pipeline
- Advanced unsupervised anomaly detection
- Deep learning (autoencoders, LSTM for sequential transactions)
- Production-grade REST API for scoring
- Cloud deployment (AWS/GCP/Azure) with model monitoring

---

## Viva Questions & Answers

1. **What is fraud detection?** Identifying transactions that are illegitimate
   or unauthorized using data patterns.
2. **What is imbalanced data?** A dataset where one class (genuine) vastly
   outnumbers the other (fraud).
3. **Why is accuracy misleading here?** Predicting "genuine" always still
   gives ~99% accuracy while catching zero fraud.
4. **What is SMOTE?** A technique that creates synthetic minority samples by
   interpolating between existing minority points, instead of duplicating.
5. **Why should SMOTE not be applied to test data?** It would leak
   synthetic, non-real patterns into evaluation, giving fake performance.
6. **What is precision?** Of predicted frauds, the fraction that are
   actually fraud.
7. **What is recall?** Of actual frauds, the fraction correctly caught.
8. **What is F1-score?** The harmonic mean of precision and recall.
9. **What is PR-AUC?** Area under the precision-recall curve; better than
   ROC-AUC for rare-event problems.
10. **What is a false positive?** A genuine transaction wrongly flagged as
    fraud.
11. **What is a false negative?** A fraudulent transaction wrongly passed as
    genuine — the costliest error in fraud detection.
12. **Why is recall important in fraud detection?** Missing real fraud
    (false negatives) usually costs more than a false alarm.
13. **What is class weighting?** Giving the minority class a higher penalty
    weight during training so the model doesn't ignore it.
14. **Why use XGBoost?** It handles non-linear patterns well and often gives
    the best performance on structured/tabular data.
15. **What is threshold tuning?** Choosing a probability cutoff (other than
    the default 0.5) to balance precision and recall for the business goal.
16. **What is data leakage?** When information from outside the training
    set (e.g. test data, future data) improperly influences the model.
17. **What is feature importance?** A ranking of which input features most
    influence the model's predictions.
18. **What is cross-validation?** Splitting data into multiple folds to
    validate model performance more robustly than a single split.
19. **How did you select the final model?** By comparing recall, precision,
    F1 and PR-AUC — not accuracy — and picking the model with the best
    fraud-catching ability at an acceptable false-positive rate.
20. **What are the limitations of this project?** Static dataset (no
    real-time stream), anonymized PCA features limit interpretability, and
    performance depends heavily on the dataset provided.
