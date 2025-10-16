import sys
sys.path.append("..")

from grid_env_gym.envs import GridEnv


# 1) default: init_random True -> positions set in __init__
env1 = GridEnv(M=6, N=6, K=4, seed=123, init_random=True)
print("env1 agent:", env1.agent_pos, "goal:", env1.goal_pos, "num obstacles:", len(env1.obstacles))

# 2) make sure the seed is working 
env2 = GridEnv(M=6, N=6, K=4, seed=42, init_random=True)
print("env2 agent:", env2.agent_pos, "goal:", env2.goal_pos, "num obstacles:", len(env2.obstacles))

# 3) provide explicit start_pos and init_random True
env3 = GridEnv(M=6, N=6, K=4, seed=42, init_random=True, start_pos=(0,0))
print("env3 agent (should be (0,0)):", env3.agent_pos, "goal:", env3.goal_pos, "obstacles:", env3.obstacles)


