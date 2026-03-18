# =========================================
# MODEL TRAINING FILE
# =========================================

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Dataset URLs
urls = [
    "https://raw.githubusercontent.com/SHREYAS4678/Energy-Consumption-Forecasting-/refs/heads/main/AEP_hourly.csv",
    "https://raw.githubusercontent.com/SHREYAS4678/Energy-Consumption-Forecasting-/refs/heads/main/COMED_hourly.csv",
    "https://raw.githubusercontent.com/SHREYAS4678/Energy-Consumption-Forecasting-/refs/heads/main/DAYTON_hourly.csv",
    "https://raw.githubusercontent.com/SHREYAS4678/Energy-Consumption-Forecasting-/refs/heads/main/DEOK_hourly.csv",
    "https://raw.githubusercontent.com/SHREYAS4678/Energy-Consumption-Forecasting-/refs/heads/main/DOM_hourly.csv",
    "https://raw.githubusercontent.com/SHREYAS4678/Energy-Consumption-Forecasting-/refs/heads/main/DUQ_hourly.csv",
    "https://raw.githubusercontent.com/SHREYAS4678/Energy-Consumption-Forecasting-/refs/heads/main/EKPC_hourly.csv",
    "https://raw.githubusercontent.com/SHREYAS4678/Energy-Consumption-Forecasting-/refs/heads/main/FE_hourly.csv",
    "https://raw.githubusercontent.com/SHREYAS4678/Energy-Consumption-Forecasting-/refs/heads/main/NI_hourly.csv",
    "https://raw.githubusercontent.com/SHREYAS4678/Energy-Consumption-Forecasting-/refs/heads/main/PJME_hourly.csv",
    "https://raw.githubusercontent.com/SHREYAS4678/Energy-Consumption-Forecasting-/refs/heads/main/PJMW_hourly.csv",
    "https://raw.githubusercontent.com/SHREYAS4678/Energy-Consumption-Forecasting-/refs/heads/main/PJM_Load_hourly.csv"
]

dataframes = []

for url in urls:
    df = pd.read_csv(url)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.columns = ['consumption']

    region = url.split("/")[-1].split("_")[0]
    df['region'] = region

    dataframes.append(df)

# Combine all datasets
df = pd.concat(dataframes)

# =========================================
# FEATURE ENGINEERING
# =========================================

df['hour'] = df.index.hour
df['day'] = df.index.dayofweek
df['month'] = df.index.month
df['lag_1'] = df['consumption'].shift(1)
df['lag_24'] = df['consumption'].shift(24)

df['region'] = df['region'].astype('category').cat.codes

df = df.dropna()

# =========================================
# MODEL TRAINING
# =========================================

X = df.drop('consumption', axis=1)
y = df['consumption']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = XGBRegressor(n_estimators=300)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

# Save model
joblib.dump(model, "energy_model.pkl")

print("Model saved successfully ✅")
