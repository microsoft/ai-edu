import gym
import numpy as np

env = gym.make('Blackjack-v1', sab=True)
print(env.action_space)
print(env.observation_space)
print(env.spec)
for i in range(100):
    s = env.reset()
    Episode = []
    while True:
        action = np.random.choice(2)
        next_s, r, done, _ = env.step(action)
        Episode.append((s, action, r, next_s))
        if done:
            break
        s = next_s
    print(Episode)
env.close()

