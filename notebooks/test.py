import sys
import os
csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "btc_sample.csv")
csv_path = os.path.abspath(csv_path)
env = SimpleTradingEnv(csv_path=csv_path)

from stable_baselines3 import PPO
from src.envs.trading_env import SimpleTradingEnv

env = SimpleTradingEnv()
model = PPO.load("models/ppo_baseline")

obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
