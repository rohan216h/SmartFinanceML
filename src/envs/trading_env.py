# src/envs/trading_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, df, initial_balance=10000):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance

        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(5,), dtype=np.float32  # Open, High, Low, Close, Volume
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_sales_value = 0
        self.net_worth = self.initial_balance

        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        return np.array([
            float(row['Open']),
            float(row['High']),
            float(row['Low']),
            float(row['Close']),
            float(row['Volume'])
        ], dtype=np.float32)

    def step(self, action):
        row = self.df.iloc[self.current_step]
        current_price = float(row['Close'])
        reward = 0

        if action == 1:  # Buy
            if self.balance >= current_price:
                self.shares_held += 1
                self.balance -= current_price
                self.total_shares_bought += 1
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price
                self.total_sales_value += current_price

        self.current_step += 1

        self.net_worth = self.balance + self.shares_held * current_price
        self.total_asset = self.net_worth 
        reward = self.net_worth - self.initial_balance

        done = self.current_step >= len(self.df) - 1

        return self._get_observation(), reward, done, False, {}

    def render(self):
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance}")
        print(f"Shares Held: {self.shares_held}")
        print(f"Net Worth: {self.net_worth}")
