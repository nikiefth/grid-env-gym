import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO, DQN
from utils import make_env
from envs import GridEnv

# Evaluation configuration
EVAL_MAX_EPISODE_STEPS = 50
GLOBAL_SEED = 42  # fixed seed for reproducibility

def set_global_determinism(seed: int = 42):
    """Ensure deterministic behavior across all libs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Force deterministic PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def make_grid_env(seed=None):
    """Same environment constructor as training, with deterministic seeding."""
    return GridEnv(M=6, N=6, K=5, max_steps=50, seed=seed, init_random=True)

def evaluate_model(model_path, model_type="ppo", n_episodes=50, max_episode_steps=EVAL_MAX_EPISODE_STEPS, base_seed=GLOBAL_SEED):
    """Evaluate a single RL model on the GridEnv deterministically."""

    # Build env using same wrappers as training
    env = make_env(make_grid_env, max_episode_steps=max_episode_steps)

    # Load model (without attaching env to avoid wrapper conflicts)
    if model_type.lower() == "ppo":
        model = PPO.load(model_path, env=None)
    elif model_type.lower() == "dqn":
        model = DQN.load(model_path, env=None)
    else:
        raise ValueError("model_type must be 'ppo' or 'dqn'")

    total_steps, total_rewards = [], []
    successes = 0

    set_global_determinism(base_seed)

    for ep in range(n_episodes):
        # Fixed per-episode seed (so positions are identical each run)
        episode_seed = base_seed + ep
        obs, info = env.reset(seed=episode_seed)
        ep_reward, ep_steps = 0.0, 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

            if terminated or truncated:
                # Detect success from unwrapped env state
                try:
                    unwrapped = env
                    while hasattr(unwrapped, "env"):
                        unwrapped = unwrapped.env
                    if getattr(unwrapped, "agent_pos", None) == getattr(unwrapped, "goal_pos", None):
                        successes += 1
                except Exception:
                    if isinstance(info, dict) and info.get("reason") == "reached_goal":
                        successes += 1
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

    df = pd.DataFrame(results)
    print("\nDeterministic Evaluation Results:")
    print(df)
