import gym
import numpy as np
import pandas as pd

class SimpleTradingEnv(gym.Env):
    def __init__(self, csv_path="/data/btc_sample.csv"):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.t = 0
        self.position = 0.0
        self.cash = 10000.0
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)

    def reset(self):
        self.t = 0
        self.position = 0
        self.cash = 10000.0
        return self._obs()

    def _obs(self):
        price = float(self.df.iloc[self.t]['close'])
        return np.array([price, self.position], dtype=np.float32)

    def step(self, action):
        price = float(self.df.iloc[self.t]['close'])
        act = float(np.clip(action[0], -1, 1))
        self.position += act * 0.1     # buy/sell 10% each step
        self.t += 1
        done = self.t >= len(self.df) - 1
        new_price = float(self.df.iloc[self.t]['close'])
        reward = self.position * (new_price - price)
        return self._obs(), reward, done, {}