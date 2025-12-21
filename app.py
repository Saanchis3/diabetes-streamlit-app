import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Risk Prediction App")
st.write(
    "This app predicts **diabetes risk** using a machine learning model "
    "trained in **Google Colab**."
)

# -----------------------------
# Load model & scaler
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# -----------------------------
# Inputs
# -----------------------------
st.subheader("Enter Patient Details")

age = st.number_input("Age", 1, 120, 35)

hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

bmi = st.number_input("BMI", 10.0, 70.0, 25.0)
hba1c = st.number_input("HbA1c Level (%)", 3.0, 15.0, 5.5)
glucose = st.number_input("Blood Glucose Level (mg/dL)", 50, 400, 120)

gender = st.selectbox("Gender", ["Female", "Male"])
smoking = st.selectbox("Smoking History", ["never", "former", "current"])

# -----------------------------
# Encoding (MATCHING COLAB)
# -----------------------------
hypertension = 1 if hypertension == "Yes" else 0
heart_disease = 1 if heart_disease == "Yes" else 0
gender_Male = 1 if gender == "Male" else 0

smoking_history_former = 1 if smoking == "former" else 0
smoking_history_never = 1 if smoking == "never" else 0
# current → both 0

# -----------------------------
# Feature Order (VERY IMPORTANT)
# -----------------------------
FEATURE_COLUMNS = [
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

input_data = pd.DataFrame(
    [[
        age,
        hypertension,
        heart_disease,
        bmi,
        hba1c,
        glucose,
        gender_Male,
        smoking_history_former,
        smoking_history_never
    ]],
    columns=FEATURE_COLUMNS
)

# -----------------------------
# DEBUG SECTION (WHAT YOU ASKED)
# -----------------------------
with st.expander("🔎 Model Input Debug Info"):
    st.write("**Feature Order Used:**")
    st.json(FEATURE_COLUMNS)

    st.write("**Input DataFrame:**")
    st.dataframe(input_data)

    st.write("**Input Shape:**", input_data.shape)

# -----------------------------
# Prediction
# -----------------------------
st.markdown("---")

if st.button("🔍 Predict Diabetes Risk"):
    input_scaled = scaler.transform(input_data)
    probability = model.predict_proba(input_scaled)[0][1] * 100

    if probability >= 50:
        st.error("⚠️ **High Risk of Diabetes**")
    else:
        st.success("✅ **Low Risk of Diabetes**")

    st.metric("Diabetes Probability", f"{probability:.2f}%")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Model trained in Google Colab • Deployed using Streamlit")
