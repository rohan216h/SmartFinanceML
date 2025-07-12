# src/pipeline/train_rl_agent.py

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from src.envs.trading_env import TradingEnv
import os

# Load your data
df = pd.read_csv("data/sample_stock_data.csv")

# Keep only numeric columns necessary for the environment
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Initialize environment
env = TradingEnv(df)
check_env(env)

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/ppo_trading_agent")
print("✅ Model trained and saved at models/ppo_trading_agent.zip")

