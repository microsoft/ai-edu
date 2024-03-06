import common.GridWorld_MC_Model as mc_model
from common.Algo_Dyna_Q import DynaQ
import numpy as np
import common.DrawQpi as drawQpi
import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
mpl.rcParams['axes.unicode_minus']=False

# 状态空间 = 空间宽度 x 空间高度
GridWidth, GridHeight = 9, 6
# 起点，可以多个
StartStates = [18]
# 终点，可以多个
EndStates = [8]
# 动作空间
LEFT, DOWN, RIGHT, UP  = 0, 1, 2, 3
Actions = [LEFT, DOWN, RIGHT, UP]
# 初始策略
Policy = []  # empty means random policy 0.25,0.25,0.25,0.25 in GridWorld
# 转移概率: [SlipLeft, MoveFront, SlipRight, SlipBack]
Transition = [0.0, 1.0, 0.0, 0.0]
# 每走一步的奖励值，可以是0或者-1
StepReward = 0
GoalReward = 1
SpecialReward = {
    20: -1,
    32: -1,
    34: -1,
}
# 特殊移动，用于处理类似虫洞场景
SpecialMove = {}
# 墙
Blocks = [7,11,16,25,29,41,50]

class DynaQ_test(DynaQ):
    def run(self):
        steps = np.zeros(self.episodes)
        for episode in tqdm.trange(self.episodes):                    # 分幕
            step_episode = 0
            curr_state, _ = self.env.reset()
            done = False
            while not done:  # 幕内采样
                step_episode += 1
                curr_action = self.choose_action(curr_state)    # 选择动作
                next_state, reward, done, truncated, info = self.env.step(curr_action)
                # 式（10.4.1）q(s,a) <- q(s,a) + alpha * [r + gamma * Max[q(s')] - q(s,a)]
                self.Q[curr_state, curr_action] += \
                    self.alpha * (reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[curr_state, curr_action])
                self.update_policy_average(curr_state)
                self.feed(curr_state, curr_action, reward, next_state)
                for t in range(self.planning_steps):
                    state_, action_, next_state_, reward_ = self.sample()
                    self.Q[state_, action_] += \
                        self.alpha * (reward_ + self.gamma * np.max(self.Q[next_state_, :]) - self.Q[state_, action_])
                    self.update_policy_average(state_)
                curr_state = next_state
            # end while
            steps[episode] = step_episode
        #end for
        return self.Q, steps


if __name__=="__main__":
    env = mc_model.GridWorld(
        GridWidth, GridHeight, StartStates, EndStates,  # 关于状态的参数
        Actions, Policy, Transition,                    # 关于动作的参数
        GoalReward, StepReward, SpecialReward,          # 关于奖励的参数
        SpecialMove, Blocks)                            # 关于移动的限制

    episodes = 40
    runs = 10
    plan_steps = [0, 5, 50]
    lines = ["-", "--", "-.", ":"]
    steps_all = np.zeros((len(plan_steps), episodes))
    for run in range(runs):
        for i in range(len(plan_steps)):
            s, _ = env.reset()
            behavior_policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
            ctrl = DynaQ_test(env, episodes, behavior_policy, alpha=0.1, gamma=0.95, epsilon=0.1, plan_step=plan_steps[i])
            Q, steps = ctrl.run()
            steps_all[i] += steps
            # drawQpi.drawQ(Q, (6,9), round=2, goal_state=8, end_state=Blocks)
    steps_all /= runs
    for i in range(len(plan_steps)):
        plt.plot(steps_all[i], label="n={}".format(plan_steps[i]), linestyle=lines[i])
    plt.grid()
    plt.legend()
    plt.xlabel("幕")
    plt.ylabel("每幕的步数")
    plt.show()
