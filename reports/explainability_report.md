# Phase 5 — Explainable AI Report

## Project
AI Mental Health Assessment Platform

## Purpose
This report summarizes explainability outputs for the three prediction tasks:

1. Mental Health Condition Prediction
2. Severity Prediction
3. Treatment Recommendation

## Methods Used
- Built-in feature importance
- Permutation importance
- SHAP summary plot
- SHAP local explanation
- Partial Dependence Plot

## Important Disclaimer
This model is for educational and portfolio demonstration only. It should not be used as a medical diagnosis tool.

## Mental Health Condition
Top features by permutation importance:
1. sleep_hours
2. anxiety_score
3. social_media_hours
4. mood_score
5. academic_work_pressure

## Severity Level
Top features by permutation importance:
1. stress_level
2. academic_work_pressure
3. anxiety_score
4. depression_score
5. gender_Non-binary

## Recommended Treatment
Top features by permutation importance:
1. sleep_hours
2. anxiety_score
3. academic_work_pressure
4. social_media_hours
5. depression_score


## Saved Outputs
- `reports/explainability/`
- `reports/figures/feature_importance/`
- `reports/figures/shap/`
- `reports/figures/pdp/`

## Next Phase
Phase 6 will build a reusable prediction pipeline for FastAPI and Streamlit deployment.
