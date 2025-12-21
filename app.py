import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

# ----------------------------------
# Load model & scaler
# ----------------------------------
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------------------
# Feature order (MUST MATCH COLAB)
# ----------------------------------
FEATURE_NAMES = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "gender_Male",
    "smoking_history_former",
    "smoking_history_never"
]

# ----------------------------------
# Feature explanations (DISPLAY ONLY)
# ----------------------------------
FEATURE_EXPLANATIONS = {
    "age": "Increasing age raises diabetes risk",
    "hypertension": "High blood pressure is linked with diabetes",
    "heart_disease": "Heart disease often coexists with diabetes",
    "bmi": "Higher BMI increases insulin resistance",
    "HbA1c_level": "HbA1c reflects long-term blood sugar control",
    "blood_glucose_level": "Higher glucose indicates poor regulation",
    "gender_Male": {
        0: "Female gender (lower risk in this model)",
        1: "Male gender (higher risk in this model)"
    },
    "smoking_history_former": {
        1: "Former smoker (moderate risk factor)",
        0: "Not a former smoker"
    },
    "smoking_history_never": {
        1: "Never smoked (protective factor)",
        0: "Has smoking history"
    }
}

# ----------------------------------
# App title
# ----------------------------------
st.title("🩺 Diabetes Risk Prediction App")
st.write("This app predicts **diabetes risk** using a machine learning model trained in Google Colab.")

# ----------------------------------
# User inputs
# ----------------------------------
age = st.number_input("Age", min_value=1, max_value=120, value=30)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
HbA1c = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=5.5)
blood_glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=50, max_value=400, value=120)

gender = st.selectbox("Gender", ["Female", "Male"])
smoking = st.selectbox("Smoking History", ["Never", "Former", "Current"])

# ----------------------------------
# Encode categorical values (MATCH COLAB)
# ----------------------------------
hypertension = 1 if hypertension == "Yes" else 0
heart_disease = 1 if heart_disease == "Yes" else 0

gender_Male = 1 if gender == "Male" else 0
smoking_history_former = 1 if smoking == "Former" else 0
smoking_history_never = 1 if smoking == "Never" else 0

# ----------------------------------
# Input validation warnings (NON-BLOCKING)
# ----------------------------------
st.subheader("⚠️ Input Check")

if bmi > 40:
    st.warning("Very high BMI increases diabetes risk significantly.")

if HbA1c >= 6.5:
    st.warning("HbA1c ≥ 6.5% is typically diagnostic of diabetes.")

if blood_glucose >= 200:
    st.warning("Blood glucose ≥ 200 mg/dL indicates high risk.")

# ----------------------------------
# Prepare input data (STRICT ORDER)
# ----------------------------------
input_data = np.array([[
    age,
    hypertension,
    heart_disease,
    bmi,
    HbA1c,
    blood_glucose,
    gender_Male,
    smoking_history_former,
    smoking_history_never
]])

input_df = pd.DataFrame(input_data, columns=FEATURE_NAMES)

# ----------------------------------
# Predict
# ----------------------------------
if st.button("🔍 Predict Diabetes Risk"):

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ **High Risk of Diabetes**\n\nProbability: **{probability*100:.2f}%**")
    else:
        st.success(f"✅ **Low Risk of Diabetes**\n\nProbability: **{probability*100:.2f}%**")

    # ----------------------------------
    # Feature importance explanation
    # ----------------------------------
    st.divider()
    st.subheader("📊 Why this prediction?")

    coefs = model.coef_[0]
    impacts = np.abs(coefs * input_scaled[0])

    explanation_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Impact": impacts
    }).sort_values(by="Impact", ascending=False).head(5)

    st.dataframe(explanation_df, use_container_width=True)

    for _, row in explanation_df.iterrows():
        feature = row["Feature"]
        if feature in FEATURE_EXPLANATIONS:
            exp = FEATURE_EXPLANATIONS[feature]
            if isinstance(exp, dict):
                value = int(input_df[feature].iloc[0])
                st.write(f"• **{exp[value]}**")
            else:
                st.write(f"• **{exp}**")

# ----------------------------------
# Medical interpretation
# ----------------------------------
st.divider()
st.subheader("🩺 Medical Interpretation")

st.markdown("""
- **BMI**: High BMI is linked to insulin resistance.
- **HbA1c**: Measures average blood sugar over the past 2–3 months.
- **Blood Glucose**: Reflects current blood sugar levels.
- **Smoking**: Increases metabolic and cardiovascular risk.
- **Hypertension & Heart Disease**: Common diabetes comorbidities.

⚠️ *This tool is for educational purposes only and not a medical diagnosis.*
""")

st.caption("Model trained in Google Colab • Deployed using Streamlit")
