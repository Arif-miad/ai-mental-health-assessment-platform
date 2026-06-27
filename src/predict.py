
from pathlib import Path
import joblib
import pandas as pd
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"

TARGET_CONFIG = {
    "condition": {
        "model_file": "condition_best_model.pkl",
        "preprocessor_file": "condition_preprocessor.pkl",
        "label": "Mental Health Condition"
    },
    "severity": {
        "model_file": "severity_best_model.pkl",
        "preprocessor_file": "severity_preprocessor.pkl",
        "label": "Severity Level"
    },
    "treatment": {
        "model_file": "treatment_best_model.pkl",
        "preprocessor_file": "treatment_preprocessor.pkl",
        "label": "Recommended Treatment"
    }
}

FEATURE_COLUMNS = [
    "age",
    "gender",
    "occupation",
    "sleep_hours",
    "sleep_quality",
    "social_media_hours",
    "academic_work_pressure",
    "physical_activity_days",
    "stress_level",
    "anxiety_score",
    "depression_score",
    "work_life_balance",
    "mood_score",
    "concentration_level",
    "social_support",
]

NUMERIC_FEATURES = [
    "age",
    "sleep_hours",
    "sleep_quality",
    "social_media_hours",
    "academic_work_pressure",
    "physical_activity_days",
    "stress_level",
    "anxiety_score",
    "depression_score",
    "work_life_balance",
    "mood_score",
    "concentration_level",
    "social_support",
]

CATEGORICAL_FEATURES = [
    "gender",
    "occupation",
]


def load_artifact(path):
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return joblib.load(path)


def load_prediction_artifacts():
    artifacts = {}

    for target_key, config in TARGET_CONFIG.items():
        model_path = MODELS_DIR / config["model_file"]
        preprocessor_path = MODELS_DIR / config["preprocessor_file"]

        artifacts[target_key] = {
            "model": load_artifact(model_path),
            "preprocessor": load_artifact(preprocessor_path),
            "label": config["label"],
        }

    return artifacts


ARTIFACTS = None


def get_artifacts():
    global ARTIFACTS
    if ARTIFACTS is None:
        ARTIFACTS = load_prediction_artifacts()
    return ARTIFACTS


def validate_input(input_data):
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data.copy()
    else:
        raise TypeError("input_data must be a dictionary or pandas DataFrame.")

    missing_columns = [col for col in FEATURE_COLUMNS if col not in input_df.columns]
    if missing_columns:
        raise ValueError(f"Missing input columns: {missing_columns}")

    input_df = input_df[FEATURE_COLUMNS].copy()

    for col in NUMERIC_FEATURES:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    if input_df[NUMERIC_FEATURES].isnull().any().any():
        bad_cols = input_df[NUMERIC_FEATURES].columns[input_df[NUMERIC_FEATURES].isnull().any()].tolist()
        raise ValueError(f"Numeric columns contain invalid values: {bad_cols}")

    for col in CATEGORICAL_FEATURES:
        input_df[col] = input_df[col].astype(str)

    return input_df


def predict_single_target(input_df, target_key):
    artifacts = get_artifacts()

    if target_key not in artifacts:
        raise ValueError(f"Unknown target_key: {target_key}")

    model = artifacts[target_key]["model"]
    preprocessor = artifacts[target_key]["preprocessor"]

    X_processed = preprocessor.transform(input_df)
    prediction = model.predict(X_processed)

    result = {
        "prediction": prediction[0]
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_processed)[0]
        classes = model.classes_

        result["confidence"] = round(float(np.max(probabilities)), 4)
        result["probabilities"] = {
            str(cls): round(float(prob), 4)
            for cls, prob in zip(classes, probabilities)
        }
    else:
        result["confidence"] = None
        result["probabilities"] = None

    return result


def predict_mental_health(input_data):
    input_df = validate_input(input_data)

    condition_result = predict_single_target(input_df, "condition")
    severity_result = predict_single_target(input_df, "severity")
    treatment_result = predict_single_target(input_df, "treatment")

    return {
        "mental_health_condition": condition_result["prediction"],
        "condition_confidence": condition_result["confidence"],
        "severity": severity_result["prediction"],
        "severity_confidence": severity_result["confidence"],
        "recommended_treatment": treatment_result["prediction"],
        "treatment_confidence": treatment_result["confidence"],
        "details": {
            "condition_probabilities": condition_result["probabilities"],
            "severity_probabilities": severity_result["probabilities"],
            "treatment_probabilities": treatment_result["probabilities"],
        }
    }


def predict_batch(input_df):
    validated_df = validate_input(input_df)
    results = []

    for _, row in validated_df.iterrows():
        results.append(predict_mental_health(row.to_dict()))

    return pd.DataFrame([
        {
            "mental_health_condition": item["mental_health_condition"],
            "condition_confidence": item["condition_confidence"],
            "severity": item["severity"],
            "severity_confidence": item["severity_confidence"],
            "recommended_treatment": item["recommended_treatment"],
            "treatment_confidence": item["treatment_confidence"],
        }
        for item in results
    ])
