import common.GridWorld_DP_Model as dp_model
import common.GridWorld_MC_Model as mc_model
import common.Algo_DP_PolicyEvaluation as algoDP
import numpy as np
import common.CommonHelper as helper


# 状态空间 = 空间宽度 x 空间高度
GridWidth, GridHeight = 3, 3
# 起点，可以多个
StartStates = [0]
# 终点，可以多个
EndStates = [8]
# 动作空间
LEFT, DOWN, RIGHT, UP  = 0, 1, 2, 3
Actions = [LEFT, DOWN, RIGHT, UP]
# 转移概率: [SlipLeft, MoveFront, SlipRight, SlipBack]
Transition = [0.0, 1.0, 0.0, 0.0]
GoalReward = 0
# 每走一步的奖励值，可以是0或者-1
StepReward = -1
# 特殊奖励 from s->s' then get r, 其中 s,s' 为状态序号，不是坐标位置
SpecialReward = {
}
# 特殊移动，用于处理类似虫洞场景
SpecialMove = {
}
# 墙
Blocks = []

if __name__=="__main__":

    env = dp_model.GridWorld(
        GridWidth, GridHeight, StartStates, EndStates,  # 关于状态的参数
        Actions, Transition,                    # 关于动作的参数
        StepReward, SpecialReward,                      # 关于奖励的参数
        SpecialMove, Blocks)                            # 关于移动的限制
    behavior_policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    gamma = 1

    policy = np.zeros((GridHeight * GridWidth))/len(Actions)
    V, _ = algoDP.calculate_VQ_pi(env, behavior_policy, gamma=gamma)
    helper.print_V(V, 3, (GridHeight, GridWidth), helper.SeperatorLines.middle, "V")
    

    env = mc_model.GridWorld(
        GridWidth, GridHeight, StartStates, EndStates,  # 关于状态的参数
        Actions, Transition,                    # 关于动作的参数
        GoalReward, StepReward, SpecialReward,          # 关于奖励的参数
        SpecialMove, Blocks)                            # 关于移动的限制


    episodes = 1000
    gamma = 1.0
    alpha = 0.1
    all_path = set()
    V_all = []
    V = np.zeros(env.observation_space.n)
    C = np.zeros(env.observation_space.n)
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        trajactory = [state]
        R = [0]
        while not done:
            action = np.random.choice(Actions)
            next_state, reward, done, truncated, info = env.step(action)
            trajactory.append(next_state)
            R.append(reward)
            state = next_state
        # end while
        G = 0
        for t in range(len(trajactory)-1, -1, -1):
            s = trajactory[t]
            G = R[t] + gamma * G
            V[s] += G
            C[s] += 1
        all_path.add(str(trajactory))
        if (episode+1) % 100 == 0:
            V_all.append(V/C)
            helper.print_V(V/C, 3, (GridHeight, GridWidth), helper.SeperatorLines.middle, "{0}幕".format(episode+1))
            print("不同的走法：", len(all_path))
    V_All = np.array(V_all)
    for i in range(2,11):
        var = np.var(V_All[0:i], axis=0)
        helper.print_seperator_line(helper.SeperatorLines.middle, "前{0}个100幕快照的方差".format(i))
        print(np.round(var,3))

