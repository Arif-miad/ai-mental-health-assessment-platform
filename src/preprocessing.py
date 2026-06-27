"""
Reusable preprocessing utilities for the AI Mental Health Assessment Platform.

This module prepares the same lifestyle input features for three prediction targets:
1. mental_health_condition
2. severity
3. treatment
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET_COLUMNS = ["mental_health_condition", "severity", "treatment"]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the mental health CSV dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return input feature columns by removing target columns."""
    return [col for col in df.columns if col not in TARGET_COLUMNS]


def split_feature_types(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    """Identify numerical and categorical feature columns."""
    numerical_cols = df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numerical_cols, categorical_cols


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    return df.drop_duplicates().reset_index(drop=True)


def clip_outliers_iqr(df: pd.DataFrame, numerical_cols: List[str], factor: float = 1.5) -> pd.DataFrame:
    """Clip numerical outliers using the IQR method."""
    df = df.copy()
    for col in numerical_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def build_preprocessor(numerical_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Build sklearn preprocessing pipeline: median imputation + scaling, mode imputation + one-hot."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])


def preprocess_for_target(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    apply_outlier_clipping: bool = True,
):
    """Create train/test split and fitted preprocessor for one target."""
    if target_col not in TARGET_COLUMNS:
        raise ValueError(f"target_col must be one of {TARGET_COLUMNS}")

    clean_df = remove_duplicates(df)
    feature_cols = get_feature_columns(clean_df)
    numerical_cols, categorical_cols = split_feature_types(clean_df, feature_cols)

    if apply_outlier_clipping:
        clean_df = clip_outliers_iqr(clean_df, numerical_cols)

    X = clean_df[feature_cols]
    y = clean_df[target_col]

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = numerical_cols.copy()
    if categorical_cols:
        encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        feature_names.extend(encoder.get_feature_names_out(categorical_cols).tolist())

    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)

    return {
        "target": target_col,
        "feature_cols": feature_cols,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
    }


def save_preprocessor(preprocessor, output_path: str | Path) -> None:
    """Save fitted preprocessor with joblib."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output_path)
