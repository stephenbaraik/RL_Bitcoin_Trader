import streamlit as st
import pandas as pd
import numpy as np
import time
import os

# RL imports
from rl_bot import TrainConfig, train_agent, evaluate_agent

# --- Page Configuration ---
st.set_page_config(page_title="RL Bitcoin Trading Bot", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# --- Initialize Session State ---
if 'simulation_complete' not in st.session_state:
    st.session_state.simulation_complete = False
if 'final_history' not in st.session_state:
    st.session_state.final_history = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'cfg' not in st.session_state:
    st.session_state.cfg = None
if 'last_eval' not in st.session_state:
    st.session_state.last_eval = None

# --- Data Loading ---
def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        df = pd.read_csv(uploaded_file)
        if 'close' in df.columns:
            close = df['close']
        elif 'Close' in df.columns:
            close = df['Close']
        else:
            st.error("CSV must include a 'Close' or 'close' column.")
            return None
        series = pd.to_numeric(close, errors='coerce').dropna()
        series = series[series > 0]
        if len(series) < 100:
            st.warning("Dataset is short; training may be unstable.")
        return pd.DataFrame({'close': series.values})
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# --- RL Streamlit App ---

# --- Streamlit UI ---
st.title("🤖 RL Bitcoin Trader")
st.markdown("This app trains and evaluates a PPO agent on BTC prices. Not financial advice.")

# --- Sidebar Controls ---
st.sidebar.header("Data & Training Parameters")
uploaded_file = st.sidebar.file_uploader("Upload BTC CSV (must contain Close/close)", type=["csv"])

total_timesteps = st.sidebar.number_input("Total Timesteps", min_value=1000, max_value=5000000, value=100000, step=10000)
window_size = st.sidebar.number_input("Window Size", min_value=8, max_value=256, value=32, step=1)
fee = st.sidebar.number_input("Fee (fraction)", min_value=0.0, max_value=0.01, value=0.0005, step=0.0001, format="%f")

model_dir = 'models'
default_model_path = os.path.join(model_dir, 'ppo_btc')
model_path_input = st.sidebar.text_input("Model path (without .zip)", value=default_model_path)

col_a, col_b = st.sidebar.columns(2)
train_clicked = col_a.button("Train PPO")
eval_clicked = col_b.button("Evaluate")
save_clicked = col_a.button("Save Model")
load_clicked = col_b.button("Load Model")

# --- Load & show data ---
df = load_data(uploaded_file)
if df is not None:
    st.subheader("Price Series")
    st.line_chart(pd.DataFrame({'Close': df['close']}))

    if train_clicked:
        with st.spinner("Training PPO agent... this may take a while"):
            cfg = TrainConfig(total_timesteps=int(total_timesteps), window_size=int(window_size), fee=float(fee))
            try:
                model = train_agent(df, cfg, model_path=model_path_input)
                st.session_state.model = model
                st.session_state.cfg = cfg
                st.success("Training complete.")
            except Exception as e:
                st.error(f"Training failed: {e}")

    if load_clicked:
        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path_input)
            st.session_state.model = model
            if st.session_state.cfg is None:
                st.session_state.cfg = TrainConfig(total_timesteps=int(total_timesteps), window_size=int(window_size), fee=float(fee))
            st.success("Model loaded.")
        except Exception as e:
            st.error(f"Load failed: {e}")

    if save_clicked:
        if st.session_state.model is None:
            st.warning("No model in memory to save.")
        else:
            try:
                os.makedirs(os.path.dirname(model_path_input), exist_ok=True)
                st.session_state.model.save(model_path_input)
                st.success("Model saved.")
            except Exception as e:
                st.error(f"Save failed: {e}")

    if eval_clicked:
        if st.session_state.model is None:
            st.warning("Train or load a model first.")
        else:
            cfg = st.session_state.cfg or TrainConfig(total_timesteps=int(total_timesteps), window_size=int(window_size), fee=float(fee))
            with st.spinner("Evaluating..."):
                try:
                    results = evaluate_agent(st.session_state.model, df, cfg)
                    st.session_state.last_eval = results
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

    if st.session_state.last_eval is not None:
        res = st.session_state.last_eval
        pv = pd.Series(res['pv'])
        price = pd.Series(res['history']['price'])
        position = pd.Series(res['history']['position'])

        cols = st.columns(3)
        cols[0].metric("Total Return", f"{res['total_return']*100:,.2f}%")
        cols[1].metric("Sharpe (daily)", f"{res['sharpe']:.2f}")
        cols[2].metric("Max Drawdown", f"{res['max_dd']*100:,.2f}%")

        st.subheader("Portfolio Value")
        st.line_chart(pd.DataFrame({"Portfolio": pv}))

        st.subheader("Price and Position")
        st.line_chart(pd.DataFrame({"Price": price}))
        st.area_chart(pd.DataFrame({"Position Fraction": position}))
else:
    st.info("Upload a CSV to begin. A CSV with a single 'Close' column works.")