
import numpy as np
import gymnasium as gym
import common.Algo_DP_PolicyEvaluation as algoPI 
import common.CommonHelper as helper
import tqdm
import math

def test_MC_TD(env: gym.Env, policy, episodes: int, alpha: float, gamma: float):
    V_mc = np.zeros(env.observation_space.n)
    V_td = np.zeros(env.observation_space.n)
    C_mc = np.zeros(env.observation_space.n)
    for episode in tqdm.trange(episodes):
        curr_state, _ = env.reset()
        done = False
        Trajectory = []
        while not done:
            action = np.random.choice(env.action_space.n, p=policy[curr_state])
            next_state, reward, done, truncated, info = env.step(action)
            Trajectory.append((curr_state, reward))
            # TD update
            V_td[curr_state] += alpha * (reward + gamma * V_td[next_state] - V_td[curr_state])
            curr_state = next_state
        # end while
        # MC update
        G = 0
        for t in range(len(Trajectory)-1, -1, -1):
            s, r = Trajectory[t]
            G = r + gamma * G
            V_mc[s] += G
            C_mc[s] += 1
        # end for
    C_mc[C_mc==0] = 1 # avoid divide by zero
    V_mc = V_mc / C_mc
    return V_mc, V_td


if __name__=="__main__":
    
    env = gym.make('FrozenLake-v1', map_name = "8x8", is_slippery=True)
    env.reset(seed=5)
    behavior_policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    episodes = 10000
    alpha = 0.1
    gamma = 1.0
    V_dp, _  = algoPI.calculate_VQ_pi(env, behavior_policy, gamma)
    helper.print_V(V_dp, 5, (8,8), helper.SeperatorLines.middle, "DP")  
    """
    repeat = 10
    V_MC = np.zeros((repeat, env.observation_space.n))
    V_TD = np.zeros((repeat, env.observation_space.n))
    for i in range(repeat):
        print("i=", i)
        v_mc, v_td = test_MC_TD(env, behavior_policy, episodes, alpha, gamma)
        V_MC[i] = v_mc
        V_TD[i] = v_td
        print("MC error=", helper.Norm2Err(v_mc, V_dp))
        print("TD error=", helper.Norm2Err(v_td, V_dp))
    # helper.print_V(V_MC, 5, (4,4), helper.SeperatorLines.middle, "MC")  
    # helper.print_V(V_TD, 5, (4,4), helper.SeperatorLines.middle, "TD")  
    # print("MC error=", helper.RMSE(V_MC, V_dp))
    # print("TD error=", helper.RMSE(V_TD, V_dp))
    
    np.save("V_MC.npy",V_MC)
    np.save("V_TD.npy",V_TD)
    """    
    V_MC= np.load("V_MC.npy")
    V_TD= np.load("V_TD.npy")

    helper.print_seperator_line(helper.SeperatorLines.middle, "10次试验的结果的误差")
    print("MC error:\t", end="")
    for i in range(10):
        # print("{:0.3f}".format(helper.Norm2Err(V_MC[i], V_dp)), end=" ")
        # print("{:0.3f}".format(helper.Norm2Err01(V_MC[i], V_dp)), end=" ")
        print("{:0.4f}".format(helper.RMSE(V_MC[i], V_dp)), end="|")
    print()
    print("TD error:\t", end="")
    for i in range(10):
        # print("{:0.3f}".format(helper.Norm2Err(V_TD[i], V_dp)), end=" ")
        # print("{:0.3f}".format(helper.Norm2Err01(V_TD[i], V_dp)), end=" ")
        print("{:0.4f}".format(helper.RMSE(V_TD[i], V_dp)), end="|")
    print()

    helper.print_seperator_line(helper.SeperatorLines.middle, "MC 10次试验各个状态的方差")
    a = np.var(V_MC, axis=0)
    print(a)
    helper.print_seperator_line(helper.SeperatorLines.middle, "TD 10次试验各个状态的方差")
    b = np.var(V_TD, axis=0)
    print(b)

    helper.print_seperator_line(helper.SeperatorLines.middle, "MC 10次试验各个状态的平均方差")
    print(np.mean(a))
    helper.print_seperator_line(helper.SeperatorLines.middle, "TD 10次试验各个状态的平均方差")
    print(np.mean(b))
    helper.print_seperator_line(helper.SeperatorLines.middle, "MC/TD 10次试验各个状态的方差的比值的平均值")
    b[b==0] = 1
    print(np.mean(a/b))

    helper.print_seperator_line(helper.SeperatorLines.middle, "MC 10次试验各个状态平均值的RMSE误差")
    v_mc = np.mean(V_MC, axis=0)
    print(helper.RMSE(v_mc, V_dp))
    helper.print_seperator_line(helper.SeperatorLines.middle, "TD 10次试验各个状态平均值的RMSE误差")
    v_td = np.mean(V_TD, axis=0)
    print(helper.RMSE(v_td, V_dp))

