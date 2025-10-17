import sys
sys.path.append("..")

from grid_env_gym.envs import GridEnv
from grid_env_gym.utils import make_env

env = make_env(lambda: GridEnv(M=6, N=6, K=4, seed=1, init_random=True, start_pos=None), max_episode_steps=100)
obs, info = env.reset()
print("obs shape:", obs.shape)   # should be (36,) for a 6x6 grid
print("sample step")
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print("step ok", reward, terminated, truncated)