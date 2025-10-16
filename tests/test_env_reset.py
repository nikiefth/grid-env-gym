import sys
sys.path.append("..")

from grid_env_gym.envs import GridEnv



# 1) init_random False but provide start_pos -> agent set, other positions not set until reset()
env1 = GridEnv(M=5, N=5, K=3, seed=7, init_random=False, start_pos=(2,2))
print("env1 agent (init_random=False):", env1.agent_pos, "goal:", env1.goal_pos, "obstacles:", env1.obstacles)

# call reset to populate when init_random=False
obs, info = env1.reset()
print("after reset env1 agent:", info["agent_pos"], "goal:", info["goal_pos"], "obstacles count:", len(info["obstacles"]))
