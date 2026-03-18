# =========================================
# STREAMLIT APP
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("⚡ Energy Consumption Forecast App")

# Load model
model = joblib.load("energy_model.pkl")

# Load dataset
df = pd.read_csv("AEP_hourly.csv")
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)
df.columns = ['consumption']

# Feature engineering
df['hour'] = df.index.hour
df['day'] = df.index.dayofweek
df['month'] = df.index.month
df['lag_1'] = df['consumption'].shift(1)
df = df.dropna()

# ==============================
# USER INPUT
# ==============================

st.subheader("Enter Details")

hour = st.slider("Hour", 0, 23)
day = st.slider("Day (0=Mon)", 0, 6)
month = st.slider("Month", 1, 12)
lag_1 = st.number_input("Previous Consumption")

# ==============================
# PREDICTION
# ==============================

if st.button("Predict"):
    input_data = np.array([[hour, day, month, lag_1]])
    prediction = model.predict(input_data)
    st.success(f"⚡ Predicted Consumption: {prediction[0]:.2f}")

# ==============================
# VISUALIZATION
# ==============================

st.subheader("Recent Consumption")
st.line_chart(df['consumption'].tail(200))
