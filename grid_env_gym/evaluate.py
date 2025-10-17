# evaluate_models_df.py
import os
import argparse
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from utils import make_env
from envs import GridEnv

def evaluate_model(model_path, model_type="ppo", n_episodes=100):
    """Evaluate a single RL model on the GridEnv."""
    
    # create fresh evaluation env
    def make_grid_env():
        return GridEnv(M=6, N=6, K=5, max_steps=200, seed=None, init_random=True)
    env = make_env(make_grid_env, max_episode_steps=200)

    # load model
    if model_type.lower() == "ppo":
        model = PPO.load(model_path, env=env)
    elif model_type.lower() == "dqn":
        model = DQN.load(model_path, env=env)
    else:
        raise ValueError("model_type must be 'ppo' or 'dqn'")

    total_steps, total_rewards = [], []
    successes = 0

    for _ in range(n_episodes):
        obs, info = env.reset()
        ep_reward, ep_steps = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            if terminated or truncated:
                reason = info.get("reason") if info else None
                if reason == "reached_goal":
                    successes += 1
                else:
                    try:
                        unwrapped = env
                        while hasattr(unwrapped, "env"):
                            unwrapped = unwrapped.env
                        if getattr(unwrapped, "agent_pos", None) == getattr(unwrapped, "goal_pos", None):
                            successes += 1
                    except Exception:
                        pass
                break

        total_steps.append(ep_steps)
        total_rewards.append(ep_reward)

    env.close()
    return {
        "model_type": model_type.upper(),
        "model_path": os.path.basename(model_path),
        "avg_steps": float(np.mean(total_steps)),
        "avg_reward": float(np.mean(total_rewards)),
        "success_rate (%)": float(100.0 * successes / n_episodes)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-path", type=str, help="Path to PPO model .zip file")
    parser.add_argument("--dqn-path", type=str, help="Path to DQN model .zip file")
    parser.add_argument("--n-episodes", type=int, default=100)
    args = parser.parse_args()

    results = []

    if args.ppo_path:
        results.append(evaluate_model(args.ppo_path, model_type="ppo", n_episodes=args.n_episodes))
    if args.dqn_path:
        results.append(evaluate_model(args.dqn_path, model_type="dqn", n_episodes=args.n_episodes))

    # Convert to pandas DataFrame for easy comparison
    df = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(df)
