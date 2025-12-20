import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -------------------------------
# Load model and scaler
# -------------------------------
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.write(
    "This app predicts **diabetes risk** using medical parameters. "
    "Please enter the details carefully."
)

st.divider()

# -------------------------------
# Get feature names from scaler
# -------------------------------
feature_names = list(scaler.feature_names_in_)

st.subheader("📥 Input Medical Details")

input_values = []

for feature in feature_names:
    feature_lower = feature.lower()

    if feature_lower in ["age"]:
        val = st.number_input("Age (years)", min_value=1, max_value=120, value=30)

    elif feature_lower in ["bmi"]:
        val = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

    elif feature_lower in ["hba1c_level"]:
        val = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=5.5)

    elif feature_lower in ["blood_glucose_level"]:
        val = st.number_input(
            "Blood Glucose Level (mg/dL)", min_value=50, max_value=300, value=120
        )

    elif feature_lower in ["hypertension"]:
        val = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    elif feature_lower in ["heart_disease"]:
        val = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    elif feature_lower in ["gender"]:
        gender = st.selectbox("Gender", ["Female", "Male"])
        val = 1 if gender == "Male" else 0

    else:
        # fallback (just in case)
        val = st.number_input(feature, value=0.0)

    input_values.append(val)

# -------------------------------
# Prediction
# -------------------------------
st.divider()

if st.button("🔍 Predict Diabetes Risk"):
    try:
        input_array = np.array(input_values).reshape(1, -1)
        input_df = pd.DataFrame(input_array, columns=feature_names)

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High Risk of Diabetes\n\nProbability: **{probability:.2%}**")
        else:
            st.success(f"✅ Low Risk of Diabetes\n\nProbability: **{probability:.2%}**")

    except Exception as e:
        st.error("Something went wrong during prediction.")
        st.exception(e)

# -------------------------------
# Footer
# -------------------------------
st.divider()
st.caption("Model trained in Google Colab • Deployed using Streamlit Cloud")
