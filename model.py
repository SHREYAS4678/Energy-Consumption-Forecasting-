# =========================================
# ENERGY FORECAST MODEL
# =========================================

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("AEP_hourly.csv")

# Convert to datetime
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Set index
df.set_index('Datetime', inplace=True)

# Rename column
df.columns = ['consumption']

# =========================================
# FEATURE ENGINEERING
# =========================================

# Extract time features
df['hour'] = df.index.hour
df['day'] = df.index.dayofweek
df['month'] = df.index.month

# Lag feature
df['lag_1'] = df['consumption'].shift(1)

# Remove null values
df = df.dropna()

# =========================================
# PREPARE DATA
# =========================================

X = df.drop('consumption', axis=1)
y = df['consumption']

# Split data (important: no shuffle for time-series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# =========================================
# TRAIN MODEL
# =========================================

model = XGBRegressor()
model.fit(X_train, y_train)

# =========================================
# EVALUATION
# =========================================

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

# Save model (optional)
import joblib
joblib.dump(model, "energy_model.pkl")
