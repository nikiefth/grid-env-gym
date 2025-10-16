import sys
sys.path.append("..")

from grid_env_gym.envs import GridEnv



env = GridEnv(M=10, N=8, K=3, seed=42, max_steps=50)
obs, info = env.reset()
env.render()
for t in range(20):
    a = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(a)
    print(f"t={t} a={a} r={reward} term={terminated} trunc={truncated} info={info}")
    env.render()
    if terminated or truncated:
        break