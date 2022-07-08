import numpy as np
import gym
import Algorithm.Algo_PolicyValueFunction as algo
import common.DrawQpi as drawQ

if __name__=="__main__":
    env = gym.make('FrozenLake-v1', map_name = "4x4", is_slippery=True)
    policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    gamma = 1
    iteration = 100
    V, Q = algo.calculate_VQ_pi(env, policy, gamma, iteration)
    print("随即策略下的状态价值函数")
    print(np.reshape(np.round(V,4), (4,4)))
    print("随即策略下的动作价值函数")
    print(np.round(Q,4))
    drawQ.draw(Q, (4,4))
