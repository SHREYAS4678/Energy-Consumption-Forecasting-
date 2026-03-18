# =========================================
# STREAMLIT APP (ERROR-FREE VERSION)
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

st.set_page_config(page_title="Energy Forecast", layout="wide")

st.title("⚡ Energy Consumption Forecasting System")

# =========================================
# LOAD DATA FROM GITHUB
# =========================================

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/SHREYAS4678/Energy-Consumption-Forecasting-/refs/heads/main/AEP_hourly.csv"
    
    df = pd.read_csv(url)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.columns = ['consumption']

    # Feature engineering
    df['hour'] = df.index.hour
    df['day'] = df.index.dayofweek
    df['month'] = df.index.month
    df['lag_1'] = df['consumption'].shift(1)

    return df.dropna()

df = load_data()

# =========================================
# TRAIN MODEL INSIDE APP
# =========================================

@st.cache_resource
def train_model(df):
    X = df[['hour', 'day', 'month', 'lag_1']]
    y = df['consumption']

    model = XGBRegressor(n_estimators=100)
    model.fit(X, y)

    return model

model = train_model(df)

# =========================================
# USER INPUT
# =========================================

st.sidebar.header("Input Parameters")

hour = st.sidebar.slider("Hour", 0, 23)
day = st.sidebar.slider("Day (0=Mon)", 0, 6)
month = st.sidebar.slider("Month", 1, 12)
lag_1 = st.sidebar.number_input("Previous Consumption", value=10000)

# =========================================
# PREDICTION
# =========================================

if st.sidebar.button("Predict"):
    input_data = np.array([[hour, day, month, lag_1]])
    prediction = model.predict(input_data)
    st.success(f"⚡ Predicted Consumption: {prediction[0]:.2f}")

# =========================================
# VISUALIZATION
# =========================================

st.subheader("📊 Energy Consumption Trend")
st.line_chart(df['consumption'].tail(200))
