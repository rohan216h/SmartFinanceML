import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.envs.trading_env import TradingEnv
import gym

# === Load test data ===
df = pd.read_csv("data/AAPL.csv")
df = df.dropna()

# ✅ Keep only numeric columns (drop 'Symbol' and any other string/object type)
df = df.select_dtypes(include=["number"])

# === Set up environment ===
env = TradingEnv(df)
model = PPO.load("models/ppo_trading_agent")

# === Run agent ===
obs, _ = env.reset()
done = False
total_assets = []

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    total_assets.append(env.total_asset)

# === Plot and save performance ===
os.makedirs("reports", exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(total_assets)
plt.title("RL Agent Total Asset Over Time")
plt.xlabel("Step")
plt.ylabel("Total Asset Value ($)")
plt.grid(True)
plt.tight_layout()
plt.savefig("reports/rl_equity_curve.png")
plt.show()
