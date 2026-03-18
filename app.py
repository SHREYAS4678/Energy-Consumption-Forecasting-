# =========================================
# LOAD DATA FROM URL IN STREAMLIT
# =========================================

import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/SHREYAS4678/Energy-Consumption-Forecasting-/refs/heads/main/AEP_hourly.csv"
    
    df = pd.read_csv(url)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.columns = ['consumption']
    
    return df

df = load_data()

st.line_chart(df['consumption'])
