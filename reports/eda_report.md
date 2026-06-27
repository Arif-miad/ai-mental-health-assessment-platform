# Exploratory Data Analysis Report

## Project

**AI Mental Health Assessment Platform**

This report summarizes Phase 1 EDA for the mental health lifestyle dataset. The project will later predict three outputs from the same user input:

1. Mental health condition
2. Severity level
3. Recommended treatment

---

## Dataset Overview

- **Rows:** 500
- **Columns:** 18
- **Duplicate Records:** 0
- **Numerical Features:** 13
- **Categorical Features:** 5

### Numerical Columns

age, sleep_hours, sleep_quality, social_media_hours, academic_work_pressure, physical_activity_days, stress_level, anxiety_score, depression_score, work_life_balance, mood_score, concentration_level, social_support

### Categorical Columns

gender, occupation, mental_health_condition, severity, treatment

---

## Missing Value Summary

| Column | Missing Values |
|---|---:|
| age | 0 |
| gender | 0 |
| occupation | 0 |
| sleep_hours | 15 |
| sleep_quality | 21 |
| social_media_hours | 19 |
| academic_work_pressure | 33 |
| physical_activity_days | 20 |
| stress_level | 28 |
| anxiety_score | 22 |
| depression_score | 22 |
| work_life_balance | 18 |
| mood_score | 20 |
| concentration_level | 21 |
| social_support | 30 |
| mental_health_condition | 0 |
| severity | 0 |
| treatment | 0 |

---

## Target Distribution

### Mental Health Condition

| Class | Count |
|---|---:|
| Normal | 125 |
| Anxiety | 125 |
| Burnout | 125 |
| Depression | 125 |

### Severity

| Class | Count |
|---|---:|
| Mild | 242 |
| Moderate | 229 |
| Severe | 29 |

### Occupation

| Class | Count |
|---|---:|
| Both (Part-time work + Study) | 185 |
| Student | 171 |
| Working Professional | 144 |

---

## Top Numerical Correlations

| Feature 1 | Feature 2 | Correlation |
|---|---|---:|
| stress_level | work_life_balance | -0.732 |
| work_life_balance | concentration_level | 0.706 |
| work_life_balance | mood_score | 0.702 |
| mood_score | social_support | 0.696 |
| depression_score | social_support | -0.696 |
| stress_level | concentration_level | -0.696 |
| work_life_balance | social_support | 0.687 |
| sleep_quality | work_life_balance | 0.672 |
| mood_score | concentration_level | 0.653 |
| depression_score | mood_score | -0.652 |

---

## Generated Figures

All figures are saved in `reports/figures/`.

Main figures:

- `target_distribution.png`
- `age_distribution.png`
- `gender_distribution.png`
- `occupation_distribution.png`
- `correlation_heatmap.png`
- `stress_level_by_condition_boxplot.png`
- `anxiety_score_by_condition_boxplot.png`
- `depression_score_by_condition_boxplot.png`
- `sleep_hours_by_condition_boxplot.png`
- `mood_score_by_condition_boxplot.png`
- `social_media_hours_by_condition_boxplot.png`

---

## Initial EDA Observations

- The dataset contains lifestyle, psychological score, and social support features suitable for multiclass classification.
- The same dataset can support three prediction tasks: condition, severity, and treatment recommendation.
- Numerical scores such as stress, anxiety, depression, mood, sleep quality, and social support are important candidate predictors.
- Categorical columns such as gender and occupation need encoding before model training.
- The treatment column contains longer text labels, so it may need label encoding or mapping before model training.

---

## Next Phase

**Phase 2: Data Preprocessing**

Planned tasks:

1. Clean target labels
2. Handle missing values
3. Remove duplicates
4. Encode categorical variables
5. Scale numerical variables
6. Prepare train/test split for three targets
