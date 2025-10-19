import os
import argparse
from stable_baselines3 import DQN
from utils import make_env   # expected to return a wrapped gym.Env (Monitor + FlattenObservation + TimeLimit)
from envs import GridEnv

MODEL_DIR = "models"
TB_LOG_DIR = "tb_logs"

def make_grid_env() -> GridEnv:
    """Constructor for a single GridEnv instance."""
    return GridEnv(M=6, N=6, K=5, max_steps=50, seed=None, init_random=True)

def train(total_timesteps: int = 100_000,
          save_name: str = "dqn_grid",
          tensorboard_log_dir: str = TB_LOG_DIR):
    # ensure directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    # Create a single wrapped environment (DQN is typically trained on a single env)
    env = make_env(make_grid_env, max_episode_steps=50)

    # DQN hyperparameters tuned for small discrete grid
    policy_kwargs = dict(net_arch=[128, 128])

    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=64,
        tau=1.0,                       # soft update coefficient (1.0 means hard update)
        train_freq=4,                  # every 4 environment steps a gradient step is performed
        target_update_interval=1000,   # how often to update target network
        exploration_fraction=0.5,
        exploration_final_eps=0.02,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log_dir,
        # device="auto"  # optional: set to 'cuda' if you want GPU
    )

    tb_log_name = f"{save_name}"
    print(f"Starting DQN training: total_timesteps={total_timesteps}, tb_log_name={tb_log_name}")

    model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)

    model_path = os.path.join(MODEL_DIR, f"{save_name}.zip")
    model.save(model_path)
    print(f"Saved final DQN model to: {model_path}")

    env.close()
    return model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on GridEnv")
    parser.add_argument("--size", choices=["small", "medium", "large"], default="medium",
                        help="Training size / duration")
    args = parser.parse_args()

    if args.size == "small":
        train(total_timesteps=20_000, save_name="dqn_grid_small")
    elif args.size == "medium":
        train(total_timesteps=100_000, save_name="dqn_grid_medium")
    else:  # large
        train(total_timesteps=500_000, save_name="dqn_grid_large")
