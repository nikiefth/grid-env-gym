import sys
sys.path.append("..")

from grid_env_gym.envs import GridEnv


env = GridEnv(M=5, N=5, K=3, seed=7, max_steps=50)
obs, info = env.reset()
print("Start:\n", obs)
done = False
for t in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"t={t} action={action} reward={reward} term={terminated} trunc={truncated} info={info}")
    print(obs)
    if terminated or truncated:
        print("Episode ended:", info)
        break

print("#"*20)
# Check max steps is working
env = GridEnv(M=5, N=5, K=3, seed=7, start_pos=(2,2),max_steps=2)
obs  = env._get_observation()
print("Start:\n", obs)
done = False
for t in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"t={t} action={action} reward={reward} term={terminated} trunc={truncated} info={info}")
    print(obs)
    if terminated or truncated:
        print("Episode ended:", info)
        break