import warnings

import numpy as np
import pandas as pd
import streamlit as st
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Disease Risk Predictor", layout="wide")
st.title("Health Disease Risk Prediction Dashboard")


@st.cache_data
def load_and_prepare_data(path: str = "data/LLCP2024.XPT"):
    df = pd.read_sas(path, format="xport", encoding="latin1")

    target_indicators = ["DIABETE4", "CVDINFR4", "CVDCRHD4", "CVDSTRK3"]
    for col in target_indicators:
        if col in df.columns:
            df.loc[df[col].isin([7, 9]), col] = np.nan

    df = df.dropna(subset=[c for c in target_indicators if c in df.columns])

    available_target_cols = [c for c in target_indicators if c in df.columns]
    if len(available_target_cols) == 0:
        raise ValueError("Required BRFSS target columns are not present in the dataset.")

    df["DISEASE_RISK"] = (
        (df.get("DIABETE4", pd.Series(2, index=df.index)) == 1)
        | (df.get("CVDINFR4", pd.Series(2, index=df.index)) == 1)
        | (df.get("CVDCRHD4", pd.Series(2, index=df.index)) == 1)
        | (df.get("CVDSTRK3", pd.Series(2, index=df.index)) == 1)
    ).astype(int)

    candidate_features = [
        "_AGE80",
        "_BMI5",
        "GENHLTH",
        "PHYSHLTH",
        "MENTHLTH",
        "_RFSMOK3",
        "_TOTINDA",
        "_RFHLTH",
        "_RFBMI5",
        "_DRDXAR2",
    ]
    feature_cols = [col for col in candidate_features if col in df.columns]
    if len(feature_cols) < 4:
        raise ValueError("Not enough expected BRFSS feature columns found.")

    model_df = df[feature_cols + ["DISEASE_RISK"]].copy()
    model_df = model_df.fillna(model_df.median(numeric_only=True))

    X = model_df[feature_cols]
    y = model_df["DISEASE_RISK"]

    if len(model_df) > 120000:
        model_df = model_df.sample(n=120000, random_state=42)
        X = model_df[feature_cols]
        y = model_df["DISEASE_RISK"]

    ros = RandomOverSampler(random_state=42)
    X_balanced, y_balanced = ros.fit_resample(X, y)

    return X_balanced, y_balanced, feature_cols


@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=20,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


try:
    X, y, feature_cols = load_and_prepare_data()
    model, X_train, X_test, y_train, y_test = train_model(X, y)
except Exception as error:
    st.error(f"Dashboard could not load data: {error}")
    st.stop()

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Predict", "Model Metrics", "Feature Importance"])

if page == "Overview":
    st.subheader("Project Overview")
    st.write("This dashboard predicts disease risk using BRFSS 2024 data.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Samples used", f"{len(X):,}")
    col2.metric("Features used", f"{len(feature_cols)}")
    col3.metric("Target classes", f"{y.nunique()}")

    st.write("Selected features")
    st.dataframe(pd.DataFrame({"Feature": feature_cols}), use_container_width=True)

elif page == "Predict":
    st.subheader("Make a Prediction")

    input_values = {}
    for feature in feature_cols:
        feature_min = float(X_train[feature].quantile(0.01))
        feature_max = float(X_train[feature].quantile(0.99))
        feature_default = float(X_train[feature].median())

        if feature_min == feature_max:
            input_values[feature] = st.number_input(feature, value=feature_default)
        elif feature in ["GENHLTH", "_RFSMOK3", "_TOTINDA", "_RFHLTH", "_RFBMI5", "_DRDXAR2"]:
            start = int(max(0, np.floor(feature_min)))
            end = int(max(start + 1, np.ceil(feature_max)))
            default = int(np.clip(round(feature_default), start, end))
            input_values[feature] = st.slider(feature, min_value=start, max_value=end, value=default)
        else:
            input_values[feature] = st.slider(
                feature,
                min_value=feature_min,
                max_value=feature_max,
                value=feature_default,
            )

    if st.button("Predict risk"):
        input_df = pd.DataFrame([input_values], columns=feature_cols)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.write(f"Predicted risk class: {int(prediction)}")
        st.write(f"Predicted risk probability: {probability:.2%}")
        if prediction == 1:
            st.warning("Higher disease risk predicted")
        else:
            st.success("Lower disease risk predicted")

elif page == "Model Metrics":
    st.subheader("Model Metrics")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("ROC AUC", f"{auc_score:.4f}")

    st.write("Confusion matrix")
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    st.dataframe(cm_df, use_container_width=True)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write("Classification report")
    st.dataframe(report_df, use_container_width=True)

elif page == "Feature Importance":
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame(
        {"Feature": feature_cols, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    st.dataframe(importance_df, use_container_width=True)