# anomaly_detection.py

import pandas as pd
import numpy as np
import yfinance as yf
import os
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# --------------------------------------
# CONFIG
TICKER = "AAPL"
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------------------
# 1. Load or Download Stock Data
def get_data(ticker, start, end):
    path = f"{DATA_DIR}/{ticker}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False)
        df.to_csv(path)
    df["Return"] = df["Adj Close"].pct_change()
    df.dropna(inplace=True)
    return df

# --------------------------------------
# 2. Z-Score Based Anomaly Detection
def detect_zscore_anomalies(df, threshold=3):
    df = df.copy()
    df["Z_Score"] = (df["Return"] - df["Return"].mean()) / df["Return"].std()
    df["Anomaly_Z"] = df["Z_Score"].apply(lambda x: 1 if abs(x) > threshold else 0)
    return df

# --------------------------------------
# 3. Isolation Forest Based Anomaly Detection
def detect_isolation_forest(df):
    model = IsolationForest(contamination=0.01, random_state=42)
    df = df.copy()
    df["Anomaly_IF"] = model.fit_predict(df[["Return"]])
    df["Anomaly_IF"] = df["Anomaly_IF"].apply(lambda x: 1 if x == -1 else 0)
    return df

# --------------------------------------
# 4. Plot the Anomalies
def plot_anomalies(df, method="Z"):
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["Return"], label="Return", alpha=0.6)
    
    if method == "Z":
        anomalies = df[df["Anomaly_Z"] == 1]
        plt.scatter(anomalies.index, anomalies["Return"], color="red", label="Z-score Anomaly", marker="x")
    else:
        anomalies = df[df["Anomaly_IF"] == 1]
        plt.scatter(anomalies.index, anomalies["Return"], color="purple", label="IsolationForest Anomaly", marker="o")
    
    plt.title(f"{TICKER} Return Anomalies - Method: {method}")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"reports/{TICKER}_anomalies_{method}.png")
    plt.show()

# --------------------------------------
if __name__ == "__main__":
    df = get_data(TICKER, START_DATE, END_DATE)

    # Z-Score Detection
    df_z = detect_zscore_anomalies(df)
    plot_anomalies(df_z, method="Z")

    # Isolation Forest Detection
    df_if = detect_isolation_forest(df)
    plot_anomalies(df_if, method="IF")
