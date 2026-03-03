# Disease Risk Prediction Dashboard

## Project Overview
This project predicts **health disease risk** using the CDC BRFSS 2024 dataset (`LLCP2024.XPT`).

The app is built with:
- **Python**
- **Streamlit** (interactive dashboard)
- **scikit-learn** (Random Forest model)

The model predicts a binary target:
- `0` = lower disease risk
- `1` = higher disease risk

---

## Dataset
- Source file: `data/LLCP2024.XPT`
- Dataset: CDC BRFSS (Behavioral Risk Factor Surveillance System)

---

## Features Used in Model
The dashboard uses the following BRFSS variables:

1. `_AGE80`
   - Age in years (top-coded in BRFSS format)
   - Used because age is a strong risk factor.

2. `_BMI5`
   - Body Mass Index (BMI) scaled value in BRFSS.
   - Higher BMI is often associated with higher disease risk.

3. `GENHLTH`
   - Self-reported general health status.
   - Captures overall health condition.

4. `PHYSHLTH`
   - Number of days physical health was not good.
   - Indicates physical health burden.

5. `MENTHLTH`
   - Number of days mental health was not good.
   - Adds behavioral and stress-related context.

6. `_RFSMOK3`
   - Smoking risk indicator.
   - Smoking is linked to multiple chronic diseases.

7. `_TOTINDA`
   - Physical activity indicator.
   - Low activity can increase chronic disease risk.

8. `_RFHLTH`
   - Health risk summary indicator.
   - Composite risk-related health variable.

9. `_RFBMI5`
   - BMI risk indicator.
   - Categorized obesity/weight-related risk.

10. `_DRDXAR2`
    - Arthritis diagnosis indicator.
    - Chronic condition signal relevant to overall risk.

---

## Target Variable
Target is created as `DISEASE_RISK` using these diagnosis variables:
- `DIABETE4`
- `CVDINFR4`
- `CVDCRHD4`
- `CVDSTRK3`

If any of the above indicates diagnosed disease (`== 1`), target is set to high risk (`1`), otherwise low risk (`0`).

---

## Data Processing Steps
1. Load BRFSS XPT file.
2. Handle missing values:
   - Drop all-null columns.
   - Fill numeric missing values with median.
3. Create target variable (`DISEASE_RISK`).
4. Select model features.
5. Balance class distribution using `RandomOverSampler`.
6. Train `RandomForestClassifier`.

---

## Dashboard Pages
The Streamlit dashboard provides:
- **Overview**: sample count, feature count, and selected features.
- **Predict**: interactive input and risk prediction.
- **Model Metrics**: accuracy, ROC-AUC, confusion matrix, classification report.
- **Feature Importance**: top contributors from Random Forest.

---

## How to Run
```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

---

## Notes
- This is an educational project for learning predictive modeling and dashboard deployment.
- Predictions are not medical diagnosis and should not replace clinical judgment.
