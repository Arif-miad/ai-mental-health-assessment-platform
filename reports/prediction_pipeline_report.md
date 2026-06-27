# Phase 6 — Prediction Pipeline Report

## Objective

Build a reusable prediction pipeline for the AI Mental Health Assessment Platform.

## Inputs

The pipeline expects the following input features:

['age', 'gender', 'occupation', 'sleep_hours', 'sleep_quality', 'social_media_hours', 'academic_work_pressure', 'physical_activity_days', 'stress_level', 'anxiety_score', 'depression_score', 'work_life_balance', 'mood_score', 'concentration_level', 'social_support']

## Outputs

The pipeline returns three predictions:

1. Mental Health Condition
2. Severity Level
3. Recommended Treatment

## Saved Production File

- src/predict.py

## Sample Output

```json
{
    "mental_health_condition": "Anxiety",
    "condition_confidence": 0.4209,
    "severity": "Moderate",
    "severity_confidence": 0.9583,
    "recommended_treatment": "Cognitive Behavioral Therapy (CBT), mindfulness, reduce caffeine, sleep hygiene",
    "treatment_confidence": 0.568,
    "details": {
        "condition_probabilities": {
            "Anxiety": 0.4209,
            "Burnout": 0.3735,
            "Depression": 0.1773,
            "Normal": 0.0283
        },
        "severity_probabilities": {
            "Mild": 0.0015,
            "Moderate": 0.9583,
            "Severe": 0.0402
        },
        "treatment_probabilities": {
            "Cognitive Behavioral Therapy (CBT), mindfulness, reduce caffeine, sleep hygiene": 0.568,
            "No treatment needed; maintain healthy lifestyle and regular exercise": 0.0028,
            "Psychotherapy, antidepressants (SSRIs), regular exercise, social engagement": 0.0795,
            "Rest, work-life balance improvement, therapy, reduce workload, meditation": 0.3498
        }
    }
}
```

## Next Phase

Phase 7 will build a FastAPI backend using `src/predict.py`.
