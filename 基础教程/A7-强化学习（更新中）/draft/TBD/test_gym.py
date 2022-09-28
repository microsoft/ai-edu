import gym
#env = gym.make("LunarLander-v2")
#env = gym.make("CartPole-v1")
env = gym.make("FrozenLake-v1")
env.action_space.seed(42)

observation, info = env.reset(seed=42, return_info=True)

for _ in range(100):
    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()

    if done:
        observation, info = env.reset(return_info=True)
        print("done")
        print(observation)

env.close()
