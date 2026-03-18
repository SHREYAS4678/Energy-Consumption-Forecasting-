# =========================================
# LOAD DATA FROM GITHUB URL
# =========================================

import pandas as pd

# List of dataset URLs
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
    
    # Convert datetime
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # Set index
    df.set_index('Datetime', inplace=True)
    
    # Rename column
    df.columns = ['consumption']
    
    # Add region name from URL
    region = url.split("/")[-1].split("_")[0]
    df['region'] = region
    
    dataframes.append(df)

# Combine all datasets
df = pd.concat(dataframes)

print("Data Loaded Successfully ✅")
print(df.head())
