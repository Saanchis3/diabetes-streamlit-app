import streamlit as st
import numpy as np
import joblib

# ------------------------------
# Load model and scaler
# ------------------------------
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.markdown("""
This app predicts **diabetes risk** using the **same machine learning model trained in Google Colab**.
""")

# ------------------------------
# USER INPUTS (HUMAN FRIENDLY)
# ------------------------------
age = st.number_input("Age", 1, 120, 30)
bmi = st.number_input("BMI", 10.0, 70.0, 25.0)
hba1c = st.number_input("HbA1c Level (%)", 3.0, 15.0, 5.5)
glucose = st.number_input("Blood Glucose Level (mg/dL)", 50, 350, 120)

hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

gender = st.selectbox("Gender", ["Female", "Male"])
smoking = st.selectbox(
    "Smoking History",
    ["never", "former", "current"]
)

# ------------------------------
# ENCODING (MATCH COLAB)
# ------------------------------
hypertension = 1 if hypertension == "Yes" else 0
heart_disease = 1 if heart_disease == "Yes" else 0
gender_male = 1 if gender == "Male" else 0

smoking_current = 1 if smoking == "current" else 0
smoking_former = 1 if smoking == "former" else 0
smoking_never = 1 if smoking == "never" else 0

# ------------------------------
# FINAL FEATURE VECTOR (ORDER MATTERS)
# ------------------------------
input_data = np.array([[
    age,
    bmi,
    hba1c,
    glucose,
    hypertension,
    heart_disease,
    gender_male,
    smoking_current,
    smoking_former,
    smoking_never
]])

# ------------------------------
# PREDICTION
# ------------------------------
if st.button("🔍 Predict Diabetes Risk"):
    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes\n\nProbability: {prob*100:.2f}%")
    else:
        st.success(f"✅ Low Risk of Diabetes\n\nProbability: {prob*100:.2f}%")

st.markdown("---")
st.caption("Model trained in Google Colab • Deployed using Streamlit")
