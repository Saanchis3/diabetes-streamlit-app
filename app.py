import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict diabetes risk")

st.divider()

# ---- User Inputs ----
age = st.number_input("Age", min_value=1, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=400, value=120)

hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])

st.divider()

# ---- Prediction ----
if st.button("🔍 Predict Diabetes"):
    input_data = np.array([[age, hypertension, heart_disease, bmi, hba1c, glucose]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes\n\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ Low Risk of Diabetes\n\nProbability: {probability:.2%}")
