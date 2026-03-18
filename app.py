# =========================================
# STREAMLIT APP
# =========================================

import streamlit as st
import numpy as np
import joblib
from utils import load_data

st.set_page_config(page_title="Energy Forecast", layout="wide")

st.title("⚡ Energy Consumption Forecasting System")

# Load model
model = joblib.load("energy_model.pkl")

# Load data
df = load_data()

# Sidebar
st.sidebar.header("Input Parameters")

hour = st.sidebar.slider("Hour", 0, 23)
day = st.sidebar.slider("Day (0=Mon)", 0, 6)
month = st.sidebar.slider("Month", 1, 12)
lag_1 = st.sidebar.number_input("Previous Consumption")

# Prediction
if st.sidebar.button("Predict"):
    input_data = np.array([[hour, day, month, lag_1]])
    prediction = model.predict(input_data)
    st.success(f"⚡ Predicted Consumption: {prediction[0]:.2f}")

# Charts
st.subheader("📊 Energy Consumption Trend")
st.line_chart(df['consumption'].tail(200))
