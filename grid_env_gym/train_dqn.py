import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from utils import make_env
from envs import GridEnv

MODEL_DIR = "models"
TB_LOG_DIR = "tb_logs"

def make_grid_env():
    return GridEnv(M=6, N=6, K=5, max_steps=100, seed=None, init_random=True)

def train(total_timesteps=100000, save_name="dqn_grid"):
    env = make_env(make_grid_env, max_episode_steps=100)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        tensorboard_log=TB_LOG_DIR   #enable TensorBoard logging
    )
    tb_log_name = f"{save_name}_{total_timesteps}"
    print("Starting DQN training with TensorBoard logging...")
    model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)

    final_path = os.path.join(MODEL_DIR, f"{save_name}.zip")
    model.save(final_path)
    print(f"Saved DQN model to: {final_path}")
    env.close()
    return final_path

if __name__ == "__main__":
    train(total_timesteps=500000, save_name="dqn_grid_large")