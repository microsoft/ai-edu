import numpy as np
import gym

def play_policy(env, policy, render=False):
    total_reward = 0.
    observation = env.reset(seed=5)
    while True:
        if render:
            env.render() # 此行可显示
        action = np.random.choice(env.action_space.n,
                p=policy[observation])
        observation, reward, done, _ = env.step(action)
        total_reward += reward  # 统计回合奖励
        if done: # 游戏结束
            break
    return total_reward

def policy_v2q(env,v,s=None,gamma=1.0):
    if s is not None: # 计算第v[s]
        q = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for p,next_state,reward,done in env.unwrapped.P[s][action]:
                q[action] += p*(reward + gamma*v[next_state]*(1-done))
    else:
        q = np.zeros((env.observation_space.n,env.action_space.n))
        for state in range(env.observation_space.n):
            q[state] = policy_v2q(env,v,state,gamma)
    return q

# 下面是策略评估
def policy_evluate(env,policy,gamma=1.0,tolerant=1e-4):
    v = np.zeros(env.observation_space.n)
    vs = np.zeros(env.observation_space.n)
    q = np.zeros(env.action_space.n)
    while True:
        delta = 0
        for state in range(env.observation_space.n):
            vs[state] = sum(policy_v2q(env,v,state,gamma)*policy[state])
            delta = max(delta,vs[state]-v[state])
            v[state] = vs[state]
        if delta<tolerant :
            break
    return v

# 下面是策略改进
def policy_improve(env,v,policy,gamma=1.0):
    optimal = True
    for state in range(env.observation_space.n):
        q = policy_v2q(env,v,state,gamma)
        a_new = np.argmax(q)  # 该步执行贪心策略
        if policy[state][a_new] != 1:
            policy[state] = 0
            policy[state][a_new] = 1
            optimal = False
    return optimal,policy

# 下面是进行策略迭代
def policy_iterative(env,gamma=1.0,tolerant=1e-6):
    # 下面先进行随机策略初始化
    policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    # 不断进行策略评估与策略改进
    while True:
        v = policy_evluate(env,policy,gamma,tolerant)
        optimal, policy = policy_improve(env,v,policy,gamma)
        if optimal == True:
            break
    return policy,v

if __name__=="__main__":
    # 下面尝试用随机策略玩
    env = gym.make('FrozenLake-v1', map_name = "4x4", is_slippery=True)
    random_policy = \
        np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    episode_rewards = [play_policy(env, random_policy) for _ in range(100)]
    print("随机策略 平均奖励：{:.2f}".format(np.mean(episode_rewards)))
    # 下面尝试自己编写的策略评估函数
    print('状态价值函数：')
    v_random = policy_evluate(env,random_policy)
    print(v_random.reshape(4,4))
    print('动作函数:')
    q_random = policy_v2q(env,v_random,s=None)
    print(q_random)
    #print('更新前策略是:')
    #print(random_policy)

    exit(0)

    # 下面开始进行策略改进
    optimal,policy = policy_improve(env, v_random, random_policy, gamma=1.0)
    # 更新前策略是:
    print('在一次策略改进之后:')
    if optimal == True:
        print('找到最优解,策略是:')
        print(policy)
    else:
        print('未找到最优解，策略是:')
        print(policy)

    # 下面是完整代码
    print('*****************下面是完整策略迭代结果********************')
    policy,v = policy_iterative(env, gamma=1.0, tolerant=1e-6)
    print('找到的最优策略是:{}'.format(policy))
    print('找到最优策略是')
    print(np.argmax(policy,axis=1).reshape(4,4))
  
