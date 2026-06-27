"""Phase 4 model evaluation utilities."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
    classification_report, roc_auc_score, log_loss,
)
from sklearn.preprocessing import label_binarize


def safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:
            return None
    return None


def calculate_multiclass_roc_auc(y_true, y_proba, classes):
    try:
        if y_proba is None:
            return np.nan
        if len(classes) == 2:
            return roc_auc_score(y_true, y_proba[:, 1])
        y_true_bin = label_binarize(y_true, classes=classes)
        return roc_auc_score(y_true_bin, y_proba, average="weighted", multi_class="ovr")
    except Exception:
        return np.nan


def evaluate_classifier(model, X_test, y_test, target_name="target"):
    y_pred = model.predict(X_test)
    y_proba = safe_predict_proba(model, X_test)
    classes = np.array(sorted(pd.Series(y_test).unique()))

    metrics = {
        "target": target_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "roc_auc_weighted_ovr": calculate_multiclass_roc_auc(y_test, y_proba, classes),
    }

    try:
        metrics["log_loss"] = log_loss(y_test, y_proba, labels=classes) if y_proba is not None else np.nan
    except Exception:
        metrics["log_loss"] = np.nan

    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).T

    predictions_df = pd.DataFrame({
        "actual": pd.Series(y_test).reset_index(drop=True),
        "predicted": pd.Series(y_pred).reset_index(drop=True),
    })
    predictions_df["is_correct"] = predictions_df["actual"] == predictions_df["predicted"]

    return metrics, predictions_df, report_df
