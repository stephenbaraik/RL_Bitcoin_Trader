from stable_baselines3 import PPO
from envs.trading_env import SimpleTradingEnv

if __name__ == "__main__":
    env = SimpleTradingEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000)
    model.save("/models/ppo_baseline")
    print("Model saved to /models/ppo_baseline")