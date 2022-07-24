import gym

env = gym.make('Blackjack-v1', sab=True)
print(env.action_space)
print(env.observation_space)
