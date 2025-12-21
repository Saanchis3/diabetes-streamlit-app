import streamlit as st
import numpy as np
import joblib

# -------------------------------
# Load model & scaler
# -------------------------------
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURES = list(scaler.feature_names_in_)  # EXACT COLAB FEATURES

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("🩺 Diabetes Risk Prediction")

st.markdown("This app uses the **same model and scaler trained in Google Colab**.")

# -------------------------------
# USER INPUTS (HUMAN)
# -------------------------------
age = st.number_input("Age", 1, 120, 30)
bmi = st.number_input("BMI", 10.0, 70.0, 25.0)
hba1c = st.number_input("HbA1c Level (%)", 3.0, 15.0, 5.5)
glucose = st.number_input("Blood Glucose Level (mg/dL)", 50, 350, 120)

hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
gender = st.selectbox("Gender", ["Female", "Male"])
smoking = st.selectbox("Smoking History", ["never", "former", "current"])

# -------------------------------
# ENCODING (MATCH COLAB)
# -------------------------------
row = {
    "age": age,
    "bmi": bmi,
    "HbA1c_level": hba1c,
    "blood_glucose_level": glucose,
    "hypertension": 1 if hypertension == "Yes" else 0,
    "heart_disease": 1 if heart_disease == "Yes" else 0,
    "gender_Male": 1 if gender == "Male" else 0,
    "smoking_history_current": 1 if smoking == "current" else 0,
    "smoking_history_former": 1 if smoking == "former" else 0,
    "smoking_history_never": 1 if smoking == "never" else 0,
}

# -------------------------------
# BUILD INPUT IN EXACT ORDER
# -------------------------------
input_data = np.array([[row[col] for col in FEATURES]])

# -------------------------------
# DEBUG (OPTIONAL – KEEP FOR NOW)
# -------------------------------
st.write("Feature order:", FEATURES)
st.write("Input shape:", input_data.shape)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔍 Predict Diabetes Risk"):
    input_scaled = scaler.transform(input_data)
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes\n\nProbability: {probability*100:.2f}%")
    else:
        st.success(f"✅ Low Risk of Diabetes\n\nProbability: {probability*100:.2f}%")

st.markdown("---")
st.caption("Model trained in Google Colab • Deployed using Streamlit")
