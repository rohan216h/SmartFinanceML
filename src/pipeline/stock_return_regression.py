# stock_return_regression.py

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# ------------------------------------------
# CONFIG
TICKER = "AAPL"
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------------------
# 1. Download Historical Stock Data
def fetch_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.to_csv(f"{DATA_DIR}/{ticker}.csv")
    return df

# ------------------------------------------
# 2. Feature Engineering
def create_features(df):
    df = df.copy()
    df["Return"] = df["Adj Close"].pct_change()
    df["Lag_1"] = df["Return"].shift(1)
    df["Lag_2"] = df["Return"].shift(2)
    df["Lag_3"] = df["Return"].shift(3)
    df.dropna(inplace=True)
    return df

# ------------------------------------------
# 3. Model Training & Evaluation
def train_models(df):
    X = df[["Lag_1", "Lag_2", "Lag_3"]]
    y = df["Return"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # XGBoost
    xgb = XGBRegressor(objective="reg:squarederror", n_estimators=100)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    # Evaluation
    print("Linear Regression:")
    print(f"  RMSE: {mean_squared_error(y_test, y_pred_lr, squared=False):.5f}")
    print(f"  R^2:  {r2_score(y_test, y_pred_lr):.5f}")
    
    print("\nXGBoost:")
    print(f"  RMSE: {mean_squared_error(y_test, y_pred_xgb, squared=False):.5f}")
    print(f"  R^2:  {r2_score(y_test, y_pred_xgb):.5f}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="True Returns", alpha=0.6)
    plt.plot(y_pred_lr, label="Linear Regression", linestyle='--')
    plt.plot(y_pred_xgb, label="XGBoost", linestyle='--')
    plt.title(f"{TICKER} Return Prediction")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"reports/{TICKER}_return_prediction.png")
    plt.show()

# ------------------------------------------
if __name__ == "__main__":
    df = fetch_stock_data(TICKER, START_DATE, END_DATE)
    df_feat = create_features(df)
    train_models(df_feat)
