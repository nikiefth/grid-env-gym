import sys
sys.path.append("..")

from grid_env_gym.envs import GridEnv

env = GridEnv(M=6, N=8, K=8, seed=42)
obs, info = env.reset()
env.render()
print("Agent at:", info['agent_pos'], "Goal at:", info['goal_pos'])