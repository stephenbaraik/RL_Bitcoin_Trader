### Project Report & Implementation Template

Name: Stephen Baraik

Roll number: D-54

Program: B.Sc. (Data Science)

Semester: V , Technology Lab - III

### 1. Project Title

Reinforcement Learning Bitcoin Trading Bot using PPO and Streamlit

### 2. Abstract

This project develops a reinforcement learning (RL) agent to trade Bitcoin using Proximal Policy Optimization (PPO). A custom trading environment converts historical prices into observations representing recent normalized price changes and the agent’s current position. The action is a continuous allocation fraction to Bitcoin, and the reward is the relative change in portfolio value after transaction fees. A Streamlit app enables users to upload a CSV dataset, train the PPO model, evaluate performance, and visualize portfolio value, price, and position over time. Results indicate that RL can learn adaptive allocation strategies on historical data, but performance depends on data quality, hyperparameters, and market regimes. The work demonstrates an end-to-end RL workflow for financial decision-making and outlines future directions such as richer features, Gymnasium migration, and risk-aware objectives.

### 3. Introduction

- **Background**: Trading is a sequential decision problem where actions influence future rewards. RL is well-suited for such tasks.
- **Importance**: Automated strategies can respond consistently to market changes and reduce human bias.
- **Applications**: Crypto trading, portfolio allocation, quant research, robo-advisory.

### 4. Problem Statement & Objectives

- **Problem**: Learn a policy to dynamically allocate a portfolio between cash and Bitcoin to maximize risk-adjusted returns, accounting for trading fees.
- **Objectives**:
  - Build a stable custom trading environment with robust observations and rewards.
  - Train a PPO agent on historical closing prices.
  - Provide a Streamlit UI for dataset upload, train/evaluate, and model save/load.
  - Visualize portfolio value, price, and position fraction; report return, Sharpe, and max drawdown.

### 5. Literature Review (Optional)

- Schulman et al., Proximal Policy Optimization (PPO) Algorithms.
- Moody & Saffell (1999), Reinforcement Learning for Trading.
- Stable-Baselines3 applications in financial RL.

### 6. Dataset Description

- **Source**: Public BTC-USD price data (e.g., Yahoo Finance) or Kaggle crypto datasets.
- **Link (example)**: `https://finance.yahoo.com/quote/BTC-USD/history/` (export CSV)
- **Local sample**: `data/btc_sample.csv`
- **Records & features**: Varies by export; minimum required column: `Close` or `close`.
- **Data type**: Numeric time series.
- **Sample (first 5 rows)**: Use `pd.read_csv(...).head()`; app expects a `Close/close` column.

### 7. Methodology

1. Load CSV; sanitize: drop NaNs, keep positive prices; map to `close` column.
2. Observation: recent window of normalized price relatives + current position fraction.
3. Action space: continuous in [0, 1] representing target BTC allocation of portfolio.
4. Execute trades with fee; compute reward as relative change in portfolio value.
5. Train PPO (Stable-Baselines3) for configured timesteps.
6. Evaluate deterministically; compute total return, Sharpe ratio, and max drawdown.
7. Visualize outputs in Streamlit (portfolio value, price, position).

- **Algorithm**: PPO (clipped policy gradient).

### 8. Tools & Software Used

- **Programming Language**: Python
- **Platform**: Local (Streamlit); optional Google Colab for experiments
- **Libraries**: pandas, numpy, torch, stable-baselines3, streamlit, gym/gymnasium, matplotlib/plotly (optional)

### 9. Implementation (Code)

- Repository provides an interactive app (`app.py`) that wires PPO training/evaluation to `rl_bot.py`.
- Quick start locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

- Google Colab minimal example (replace dataset path as needed):

```python
# Step 1: Install
!pip install pandas numpy torch stable-baselines3 gymnasium matplotlib -q

# Step 2: Imports
import pandas as pd
import numpy as np
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# Step 3: Config & Env (simplified)
@dataclass
class TrainConfig:
    total_timesteps: int = 100_000
    window_size: int = 32
    fee: float = 0.0005

class BTCTradingEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, df, window_size=32, init_cash=10_000.0, fee=0.0005):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.init_cash = init_cash
        self.fee = fee
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size + 1,), dtype=np.float32)
        self.reset()
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash = float(self.init_cash)
        self.holdings = 0.0
        return self._get_obs(), {}
    def _get_obs(self):
        start = self.current_step - self.window_size + 1
        window = self.df['close'].iloc[start:self.current_step + 1].values.astype(np.float32)
        window = np.where(np.isnan(window) | (window <= 0), 1e-6, window)
        cur = window[-1]
        rel = (window / cur) - 1.0
        pos = np.array([self.get_pos()], dtype=np.float32)
        return np.nan_to_num(np.concatenate([rel, pos]).astype(np.float32))
    def get_pos(self):
        price = float(self.df['close'].iloc[self.current_step])
        price = price if (price > 0 and not np.isnan(price)) else 1e-6
        pv = self.cash + self.holdings * price
        return 0.0 if (pv <= 0 or np.isnan(pv)) else float((self.holdings * price) / pv)
    def step(self, action):
        target = float(np.clip(action[0], 0.0, 1.0))
        cur = float(self.df['close'].iloc[self.current_step])
        cur = cur if (cur > 0 and not np.isnan(cur)) else 1e-6
        nxt = float(self.df['close'].iloc[min(self.current_step + 1, len(self.df)-1)])
        nxt = nxt if (nxt > 0 and not np.isnan(nxt)) else cur
        holdings_val = self.holdings * cur
        prev_pv = max(self.cash + holdings_val, 1e-6)
        desired_val = target * prev_pv
        trade_val = desired_val - holdings_val
        fees = abs(trade_val) * 0.0005
        self.holdings += (trade_val / cur)
        self.cash += (-trade_val - fees)
        new_pv = self.cash + self.holdings * nxt
        reward = (new_pv - prev_pv) / (prev_pv + 1e-12)
        self.current_step += 1
        done = (self.current_step >= len(self.df) - 1)
        return self._get_obs(), float(reward), done, False, {"portfolio_value": new_pv}

# Step 4: Data
df = pd.read_csv('/content/btc.csv')
if 'Close' in df.columns and 'close' not in df.columns:
    df = df.rename(columns={'Close': 'close'})
df = df[['close']].dropna()
df = df[df['close'] > 0]

# Step 5: Train
cfg = TrainConfig()
env = BTCTradingEnv(df, window_size=cfg.window_size, fee=cfg.fee)
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=cfg.total_timesteps)

# Step 6: Evaluate
obs, _ = env.reset()
pvs = []
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, r, done, trunc, info = env.step(action)
    pvs.append(info['portfolio_value'])
    if done:
        break
plt.plot(pvs); plt.title('Portfolio Value'); plt.show()
```

### 10. Sample Output Screens

- Dataset preview (first 5 rows in Colab or Streamlit)
- Streamlit charts: price series, portfolio value, position fraction
- PPO training logs (KL, entropy, explained variance)

### 11. Results & Discussion

- The PPO agent learns allocation behavior that can improve returns on some datasets.
- Risk metrics (Sharpe, max drawdown) contextualize performance beyond raw returns.
- Results are sensitive to window size, training length, fees, and market regime; robust validation is required.

### 12. Conclusion

- **Key takeaways**: RL provides a flexible framework for sequential trading decisions; PPO is practical and stable.
- **Real-world application**: Prototype allocation overlays; backtesting research.
- **Limitations**: Non-stationary markets, transaction cost realism, feature sparsity, overfitting risk.
- **Future work**: Add technical/volume features, Gymnasium migration, risk-aware rewards, walk-forward validation.

### 13. References

- Schulman et al., Proximal Policy Optimization Algorithms.
- Stable-Baselines3 Documentation: `https://stable-baselines3.readthedocs.io/`
- Farama Gymnasium: `https://gymnasium.farama.org/`
- Yahoo Finance BTC-USD Historical Data.
