import gymnasium as gym
import numpy as np

from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.wrappers import TimeLimit

def make_env(env_ctor, *, max_episode_steps=200, seed=None):
    """
    env_ctor: callable that returns a fresh GridEnv instance (or any gym.Env)
    Returns a wrapped env suitable for SB3 training / evaluation:
      - TimeLimit (max steps)
      - FlattenObservation (so SB3 MlpPolicy gets a 1D vector)
      - RecordEpisodeStatistics (stores episode info in `info`)
    """
    env = env_ctor()
    if seed is not None:
        try:
            env.reset(seed=seed)
        except Exception:
            pass
    # enforce maximum episode length for safety
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    # Flatten the Box observation (MxN) into 1D for MLP policies
    env = FlattenObservation(env)
    # Record episode-level statistics (useful in evaluation)
    env = RecordEpisodeStatistics(env)
    return env