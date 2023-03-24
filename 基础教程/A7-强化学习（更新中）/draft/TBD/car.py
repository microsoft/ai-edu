import gym
env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42, return_info=True)
R = 0
for i in range(1000):
    env.render(mode="human")
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    #print(observation, reward, done, info)
    if done:
        observation, info = env.reset(return_info=True)
    R += reward
    if i % 10 == 0:
        print(i)
env.close()
print(R)
