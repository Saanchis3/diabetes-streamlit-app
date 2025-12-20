import streamlit as st
import joblib
import numpy as np

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# ================== LOAD MODEL ==================
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# ================== TITLE ==================
st.title("🩺 Diabetes Prediction App")
st.markdown(
    "This app predicts **diabetes risk** using medical parameters using a machine learning model."
)

st.divider()

# ================== USER INPUT ==================
st.subheader("🔍 Enter Patient Details")

age = st.number_input("Age", 1, 120, 30)

hypertension = st.selectbox(
    "Hypertension",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

heart_disease = st.selectbox(
    "Heart Disease",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

hba1c = st.number_input(
    "HbA1c Level",
    3.0, 15.0, 5.5,
    help="Average blood sugar over last 3 months"
)

glucose = st.number_input(
    "Blood Glucose Level",
    50, 400, 120
)

st.divider()

# ================== PREDICTION ==================
if st.button("🧠 Predict Diabetes Risk"):
    input_data = np.array([[
        age,
        hypertension,
        heart_disease,
        bmi,
        hba1c,
        glucose
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # ================== RESULT ==================
    st.subheader("📊 Prediction Result")

    if probability < 0.30:
        risk = "LOW RISK"
        st.success(f"✅ {risk}")
    elif probability < 0.70:
        risk = "MEDIUM RISK"
        st.warning(f"⚠️ {risk}")
    else:
        risk = "HIGH RISK"
        st.error(f"🚨 {risk}")

    st.markdown(f"**Probability of Diabetes:** `{probability:.2%}`")

    st.divider()

    # ================== EXPLANATION ==================
    st.subheader("🧾 Explanation")

    explanation = []

    if hba1c > 6.5:
        explanation.append("High HbA1c indicates poor long-term sugar control.")

    if glucose > 140:
        explanation.append("Elevated blood glucose is a strong diabetes indicator.")

    if bmi > 30:
        explanation.append("High BMI increases insulin resistance risk.")

    if hypertension == 1:
        explanation.append("Hypertension is commonly linked with diabetes.")

    if heart_disease == 1:
        explanation.append("Heart disease increases metabolic risk.")

    if age > 45:
        explanation.append("Diabetes risk increases with age.")

    if explanation:
        for point in explanation:
            st.write("•", point)
    else:
        st.write("• All values are within normal ranges.")

st.divider()

# ================== FOOTER ==================
st.caption(
    "⚕️ **Disclaimer:** This app is for educational purposes only. "
    "It does NOT replace professional medical advice."
)

