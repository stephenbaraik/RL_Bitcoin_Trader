import os
import numpy as np
import pandas as pd
import gym
from gym import spaces
from dataclasses import dataclass
from stable_baselines3 import PPO

# -----------------------------
# TrainConfig dataclass
# -----------------------------
@dataclass
class TrainConfig:
    total_timesteps: int
    window_size: int
    fee: float

# -----------------------------
# BTC Trading Environment with NaN/zero safeguards
# -----------------------------
class BTCTradingEnv(gym.Env):
    metadata = {'render.modes': []}

    def __init__(self, df: pd.DataFrame, window_size: int = 32, init_cash: float = 10000.0, fee: float = 0.0005):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.init_cash = init_cash
        self.fee = fee

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size + 1,), dtype=np.float32)

        self.current_step = None
        self.cash = None
        self.holdings = None
        self._last_portfolio_value = None

    def reset(self):
        self.current_step = self.window_size
        self.cash = float(self.init_cash)
        self.holdings = 0.0
        self._last_portfolio_value = float(self.init_cash)
        return self._get_obs()

    def _get_obs(self):
        start = self.current_step - self.window_size + 1
        window = self.df['close'].iloc[start:self.current_step + 1].values.copy()
        window = np.where(np.isnan(window) | (window <= 0), 1e-6, window)  # replace invalid prices

        current_price = window[-1]
        price_rel = (window / current_price) - 1.0

        pos = np.array([self.get_position_fraction()], dtype=np.float32)
        obs = np.concatenate([price_rel.astype(np.float32), pos])
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def get_position_fraction(self):
        current_price = float(self.df['close'].iloc[self.current_step])
        if np.isnan(current_price) or current_price <= 0:
            current_price = 1e-6

        holdings_value = self.holdings * current_price
        pv = self.cash + holdings_value
        if pv <= 0 or np.isnan(pv):
            return 0.0
        return float(holdings_value / pv)

    def _get_portfolio_value(self, price):
        if price <= 0 or np.isnan(price):
            price = 1e-6
        pv = float(self.cash + self.holdings * price)
        if np.isnan(pv) or pv <= 0:
            pv = 1e-6
        return pv

    def step(self, action):
        target = float(np.clip(action[0], 0.0, 1.0))
        cur_price = float(self.df['close'].iloc[self.current_step])
        if np.isnan(cur_price) or cur_price <= 0:
            cur_price = 1e-6

        next_price = float(self.df['close'].iloc[self.current_step + 1])
        if np.isnan(next_price) or next_price <= 0:
            next_price = cur_price

        holdings_value = self.holdings * cur_price
        prev_portfolio = self.cash + holdings_value
        if prev_portfolio <= 0 or np.isnan(prev_portfolio):
            prev_portfolio = 1e-6

        desired_holdings_value = target * prev_portfolio
        trade_value = desired_holdings_value - holdings_value
        fees = abs(trade_value) * self.fee

        if trade_value >= 0:
            units = trade_value / cur_price
            self.holdings += units
            self.cash -= (trade_value + fees)
        else:
            units = trade_value / cur_price
            self.holdings += units
            self.cash += (-trade_value - fees)

        new_portfolio = self._get_portfolio_value(next_price)
        reward = (new_portfolio - prev_portfolio) / (prev_portfolio + 1e-12)

        self.current_step += 1
        done = (self.current_step >= len(self.df) - 1)

        info = {
            'portfolio_value': new_portfolio,
            'cash': self.cash,
            'holdings': self.holdings,
            'cur_price': cur_price,
            'next_price': next_price,
            'fees': fees,
        }

        self._last_portfolio_value = new_portfolio
        return self._get_obs() if not done else np.zeros_like(self._get_obs()), float(reward), bool(done), info

# -----------------------------
# Training and Evaluation Functions
# -----------------------------

def train_agent(df, cfg: TrainConfig, model_path='models/ppo_btc'):
    env = BTCTradingEnv(df, window_size=cfg.window_size, fee=cfg.fee)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=cfg.total_timesteps)

    # Ensure folder exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    return model


def evaluate_agent(model, df, cfg: TrainConfig, render_plot=False):
    env = BTCTradingEnv(df, window_size=cfg.window_size, fee=cfg.fee)
    obs = env.reset()
    done = False

    history = {'price': [], 'position': []}
    pv_list = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        history['price'].append(info['cur_price'])
        history['position'].append(env.get_position_fraction())
        pv_list.append(info['portfolio_value'])

    total_return = (pv_list[-1] - pv_list[0]) / pv_list[0]
    returns = np.diff(pv_list) / (np.array(pv_list[:-1]) + 1e-12)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-12) * np.sqrt(252)
    max_dd = (np.maximum.accumulate(pv_list) - pv_list) / np.maximum.accumulate(pv_list)

    return {
        'pv': pv_list,
        'history': history,
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': np.max(max_dd)
    }