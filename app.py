import streamlit as st
import joblib

st.title("Diabetes Prediction App")

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.success("Model and scaler loaded successfully!")