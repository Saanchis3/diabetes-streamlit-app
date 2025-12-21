import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Diabetes Risk Prediction", layout="centered")

# -------------------- LOAD MODEL --------------------
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# EXACT feature order from your Colab
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

# -------------------- TITLE --------------------
st.title("🩺 Diabetes Risk Prediction App")
st.write("This app predicts **diabetes risk** using medical parameters.")

st.divider()

# -------------------- INPUTS --------------------
age = st.number_input("Age (years)", 1, 120, 30)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

bmi = st.number_input("BMI", 10.0, 70.0, 22.0)
hba1c = st.number_input("HbA1c Level (%)", 3.0, 15.0, 5.5)
glucose = st.number_input("Blood Glucose Level (mg/dL)", 50, 400, 120)

gender = st.selectbox("Gender", ["Female", "Male"])
smoking = st.selectbox("Smoking History", ["Never", "Former", "Current"])

# -------------------- ENCODING (MATCH COLAB) --------------------
gender_male = 1 if gender == "Male" else 0
smoking_former = 1 if smoking == "Former" else 0
smoking_never = 1 if smoking == "Never" else 0

# -------------------- INPUT VALIDATION --------------------
warnings = []

if bmi < 18.5 or bmi > 35:
    warnings.append("⚠️ BMI is outside the healthy range (18.5–24.9).")

if hba1c >= 5.7:
    warnings.append("⚠️ HbA1c ≥ 5.7% indicates prediabetes/diabetes risk.")

if glucose >= 140:
    warnings.append("⚠️ Blood glucose ≥ 140 mg/dL is considered high.")

if warnings:
    for w in warnings:
        st.warning(w)

# -------------------- INPUT DATAFRAME --------------------
input_data = pd.DataFrame([[
    age,
    1 if hypertension == "Yes" else 0,
    1 if heart_disease == "Yes" else 0,
    bmi,
    hba1c,
    glucose,
    gender_male,
    smoking_former,
    smoking_never
]], columns=FEATURE_NAMES)

# -------------------- PREDICTION --------------------
if st.button("🔍 Predict Diabetes Risk"):
    input_scaled = scaler.transform(input_data)
    probability = model.predict_proba(input_scaled)[0][1]

    # -------------------- RESULT --------------------
    if probability >= 0.5:
        st.error(f"🚨 **High Risk of Diabetes**\n\nProbability: **{probability*100:.2f}%**")
    else:
        st.success(f"✅ **Low Risk of Diabetes**\n\nProbability: **{probability*100:.2f}%**")

    st.divider()

    # -------------------- MEDICAL INTERPRETATION --------------------
    st.subheader("🧠 Medical Interpretation")

    if bmi >= 30:
        st.write("• **BMI** indicates obesity, a strong diabetes risk factor.")
    elif bmi >= 25:
        st.write("• **BMI** indicates overweight, increasing insulin resistance.")
    else:
        st.write("• **BMI** is in the healthy range.")

    if hba1c >= 6.5:
        st.write("• **HbA1c ≥ 6.5%** is diagnostic of diabetes.")
    elif hba1c >= 5.7:
        st.write("• **HbA1c** suggests prediabetes.")
    else:
        st.write("• **HbA1c** is normal.")

    if glucose >= 200:
        st.write("• **Blood glucose ≥ 200 mg/dL** is diabetic range.")
    elif glucose >= 140:
        st.write("• **Blood glucose** is elevated.")
    else:
        st.write("• **Blood glucose** is normal.")

    # -------------------- FEATURE IMPORTANCE (EXPLANATION) --------------------
    st.divider()
    st.subheader("📊 Why this prediction?")

    importance_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Impact": np.abs(input_scaled[0])
    }).sort_values(by="Impact", ascending=False)

    st.write("Top contributing factors for this prediction:")
    st.dataframe(importance_df.head(5), use_container_width=True)

    st.caption("Higher impact values indicate stronger influence on prediction.")

st.divider()
st.caption("Model trained in Google Colab • Deployed using Streamlit")
