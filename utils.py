# =========================================
# UTILITY FILE
# =========================================

import pandas as pd

def load_data():
    url = "https://raw.githubusercontent.com/SHREYAS4678/Energy-Consumption-Forecasting-/refs/heads/main/AEP_hourly.csv"
    
    df = pd.read_csv(url)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.columns = ['consumption']

    df['hour'] = df.index.hour
    df['day'] = df.index.dayofweek
    df['month'] = df.index.month
    df['lag_1'] = df['consumption'].shift(1)

    return df.dropna()
