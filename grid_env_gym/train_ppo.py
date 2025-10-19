# train_ppo.py
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from utils import make_env   # your wrapper helper: make_env(env_ctor, max_episode_steps=..., seed=...)
from envs import GridEnv

MODEL_DIR = "models"
TB_LOG_DIR = "tb_logs"

def make_grid_env() -> GridEnv:
    """Constructor for a single GridEnv instance."""
    return GridEnv(M=6, N=6, K=5, max_steps=50, seed=42, init_random=False)

def make_vec_env(n_envs: int = 8, max_episode_steps: int = 50) -> DummyVecEnv:
    """
    Create a DummyVecEnv with n_envs copies of the wrapped environment.
    make_env(...) is expected to return a wrapped gym.Env (Monitor + FlattenObservation + TimeLimit etc).
    """
    def _thunk():
        # return a fresh wrapped env
        return make_env(make_grid_env, max_episode_steps=max_episode_steps)
    return DummyVecEnv([_thunk for _ in range(n_envs)])

def train(total_timesteps: int = 500_000,
          save_name: str = "ppo_grid_large",
          n_envs: int = 2,
          n_steps_per_env: int = 32,
          tensorboard_log_dir: str = TB_LOG_DIR):
    # ensure directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    # create vectorized environments
    env = make_vec_env(n_envs=n_envs, max_episode_steps=50)

    # policy architecture and PPO hyperparams tuned for small grid tasks
    policy_kwargs = dict(net_arch=[128, 128])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=1e-4,
        ent_coef=0.1,        # higher entropy for better exploration
        n_steps=n_steps_per_env,
        batch_size=64,
        n_epochs=10,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log_dir,
        gamma=0.99,          # discount factor
        gae_lambda=0.95,     # GAE smoothing
        clip_range=0.2
    )

    tb_log_name = f"{save_name}"
    print(f"Starting PPO training: total_timesteps={total_timesteps}, n_envs={n_envs}, "
          f"n_steps_per_env={n_steps_per_env}, tb_log_name={tb_log_name}")

    model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)

    model_path = os.path.join(MODEL_DIR, f"{save_name}.zip")
    model.save(model_path)
    print(f"Saved final PPO model to: {model_path}")

    env.close()
    return model_path
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train PPO on GridEnv")
    parser.add_argument("--size", choices=["small", "medium", "large"], default="medium",
                        help="Training size / duration")
    args = parser.parse_args()

    if args.size == "small":
        train(total_timesteps=20_000, save_name="ppo_grid_small", n_envs=1, n_steps_per_env=64)
    elif args.size == "medium":
        train(total_timesteps=100_000, save_name="ppo_grid_medium", n_envs=2, n_steps_per_env=64)
    else:  # large
        train(total_timesteps=500_000, save_name="ppo_grid_large", n_envs=2, n_steps_per_env=64)