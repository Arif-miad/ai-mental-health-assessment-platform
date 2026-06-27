# Phase 4 Model Evaluation Report

This report summarizes the evaluation of the three trained models.

## Evaluation Metrics

| target    | display_name            |   accuracy |   balanced_accuracy |   precision_weighted |   recall_weighted |   f1_weighted |   roc_auc_weighted_ovr |   log_loss |
|:----------|:------------------------|-----------:|--------------------:|---------------------:|------------------:|--------------:|-----------------------:|-----------:|
| condition | Mental Health Condition |       0.96 |              0.96   |               0.9608 |              0.96 |        0.9602 |                 0.9949 |     0.4558 |
| severity  | Severity Level          |       0.92 |              0.7977 |               0.9171 |              0.92 |        0.9169 |                 0.9647 |     0.3164 |
| treatment | Recommended Treatment   |       0.96 |              0.96   |               0.9608 |              0.96 |        0.9602 |                 0.9971 |     0.188  |


## Generated Files

- `reports/evaluation/evaluation_metrics_summary.csv`
- `reports/evaluation/*_classification_report.csv`
- `reports/evaluation/*_predictions.csv`
- `reports/evaluation/*_class_error_analysis.csv`
- `reports/evaluation/*_confusion_pairs.csv`
- `reports/evaluation/*_misclassified_samples.csv`
- `reports/evaluation/*_feature_importance.csv`
- `reports/figures/*_confusion_matrix.png`
- `reports/figures/*_learning_curve.png`
- `reports/figures/*_feature_importance.png`


## Notes

- Accuracy shows overall correctness.
- Weighted F1 is useful when classes are imbalanced.
- Confusion matrix helps identify which classes are confused.
- Learning curve helps check overfitting and underfitting.
- Feature importance gives early insight before Phase 5 Explainable AI.
