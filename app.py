import streamlit as st
import joblib
import numpy as np

# -------------------------------
# Load model & scaler
# -------------------------------
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title("🩺 Diabetes Prediction App")

st.markdown("""
This app predicts **diabetes risk** using medical parameters.
Please enter patient details below.
""")

# -------------------------------
# Input fields (MATCH TRAINING DATA)
# -------------------------------

gender = st.selectbox("Gender", ["Female", "Male"])
smoking_history = st.selectbox(
    "Smoking History",
    ["never", "former", "current", "not current", "ever"]
)

age = st.number_input("Age", min_value=1, max_value=120, value=30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)

# -------------------------------
# Encoding (MUST MATCH TRAINING)
# -------------------------------

gender_encoded = 1 if gender == "Male" else 0

smoking_map = {
    "never": 0,
    "former": 1,
    "current": 2,
    "not current": 3,
    "ever": 4
}
smoking_encoded = smoking_map[smoking_history]

# -------------------------------
# Prediction
# -------------------------------

if st.button("🔍 Predict Diabetes Risk"):
    input_data = np.array([[
        gender_encoded,
        age,
        hypertension,
        heart_disease,
        bmi,
        hba1c,
        glucose,
        smoking_encoded
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Output
    if prediction == 1:
        st.error(f"⚠️ High risk of diabetes\n\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ Low risk of diabetes\n\nProbability: {probability:.2%}")
