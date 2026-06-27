
# Phase 3 — Model Training Report

## Project
AI Mental Health Assessment Platform

## Goal
Train 3 separate machine learning models from the same lifestyle dataset.

## Targets
1. Mental Health Condition
2. Severity Level
3. Recommended Treatment

## Best Baseline Models
- Condition: Logistic Regression
- Severity: Logistic Regression
- Treatment: Logistic Regression

## Final Tuned Results

| Target    | Best_Model         |   Accuracy |   Precision_Macro |   Recall_Macro |   F1_Macro |   Precision_Weighted |   Recall_Weighted |   F1_Weighted | Best_Params                                      |
|:----------|:-------------------|-----------:|------------------:|---------------:|-----------:|---------------------:|------------------:|--------------:|:-------------------------------------------------|
| condition | LogisticRegression |       0.96 |          0.960769 |       0.96     |   0.960192 |             0.960769 |              0.96 |      0.960192 | {'C': 0.01, 'max_iter': 2000, 'solver': 'lbfgs'} |
| severity  | LogisticRegression |       0.92 |          0.868056 |       0.797705 |   0.824409 |             0.917083 |              0.92 |      0.916851 | {'C': 1, 'max_iter': 2000, 'solver': 'lbfgs'}    |
| treatment | LogisticRegression |       0.96 |          0.960769 |       0.96     |   0.960192 |             0.960769 |              0.96 |      0.960192 | {'C': 0.1, 'max_iter': 2000, 'solver': 'lbfgs'}  |

## Saved Model Files
- `models/condition_best_model.pkl`
- `models/severity_best_model.pkl`
- `models/treatment_best_model.pkl`

## Next Phase
Phase 4: Model Evaluation with ROC-AUC, learning curves, advanced confusion matrices, and detailed error analysis.
