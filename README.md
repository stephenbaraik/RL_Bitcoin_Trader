# 🪙 RL Bitcoin Trader – Dockerized Starter Project

A beginner-friendly, containerized environment for building and experimenting with **Reinforcement-Learning-based Bitcoin trading agents**.  
Runs locally on your laptop using **Docker Compose** + **Jupyter Lab**.

---

## ✨ Features
- 🚀 **Dockerized Python 3.10** with all dependencies (Gym, Stable-Baselines3, CCXT, etc.)
- 📈 **Simple Trading Environment** (`src/envs/trading_env.py`) for quick prototyping
- 🤖 **Baseline PPO Agent** (`src/agents/train.py`) to train on sample BTC price data
- 📓 **Jupyter Lab** interface for data exploration and plotting
- 📦 **Makefile** (or direct `docker-compose` commands) for common tasks
- 📂 Volumes for **data** and **models** so they persist between container runs

---

## 📁 Project Structure
btc-rl-trader/
├─ compose/ # docker-compose configs
│ └─ docker-compose.yml
├─ docker/ # Dockerfiles
│ └─ Dockerfile.base
├─ data/ # historical BTC data (CSV / Parquet)
├─ models/ # trained models saved here
├─ notebooks/ # Jupyter notebooks
├─ src/
│ ├─ envs/trading_env.py # simple Gym env
│ ├─ agents/train.py # PPO training script
│ └─ ... # future code
├─ requirements.txt
├─ Makefile
└─ README.md

yaml
Copy code

---

## 🖥️ Prerequisites
- **Docker Desktop** (or Docker Engine + `docker-compose`)
- **Git**
- Optional: `make` utility (if unavailable, use raw docker-compose commands shown below)

---

## ⚙️ 1. Clone the Repository
```bash
git clone https://github.com/yourusername/btc-rl-trader.git
cd btc-rl-trader
📦 2. Build the Docker Image
bash
Copy code
# Using make (if installed)
make build

# OR without make
docker-compose -f compose/docker-compose.yml build
▶️ 3. Start Jupyter Lab
bash
Copy code
make up
# OR
docker-compose -f compose/docker-compose.yml up -d
Visit http://localhost:8888 to open Jupyter Lab.

🤖 4. Train the Baseline Agent
Place a sample BTC price CSV in the data/ folder (e.g., btc_sample.csv).
Then run:

bash
Copy code
make train
# OR
docker-compose -f compose/docker-compose.yml run --rm trainer
The trained PPO model will be saved under models/ppo_baseline.zip.

📊 5. Explore in Jupyter
Open a notebook in notebooks/ to:

Load and visualize BTC data

Plot training rewards

Evaluate the saved model

⏹️ 6. Stop Containers
bash
Copy code
make down
# OR
docker-compose -f compose/docker-compose.yml down
🛠️ Common Issues
make: command not found → Install make or use raw docker-compose commands.

jupyter: executable file not found → Ensure jupyterlab is in requirements.txt and rebuild.

File permission issues on Windows → Use Docker Desktop with WSL 2 backend for smoother volume mounts.

🚀 Next Steps
Add transaction costs and slippage to the reward function.

Replace sample CSV with minute-level BTC OHLCV data via the CCXT library.

Implement paper-trading executor that uses the saved PPO model.

Build a FastAPI control service to orchestrate training and paper trading.

Add monitoring (Prometheus + Grafana) for a SaaS-like setup.

📜 License
MIT (or your preferred license)

⚠️ Disclaimer:
This project is for educational and research purposes only.
Do NOT use it for live trading with real funds without extensive testing, risk controls, and regulatory compliance.