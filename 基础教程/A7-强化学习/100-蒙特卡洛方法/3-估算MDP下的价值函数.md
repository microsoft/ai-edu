## 10.3 估算 MDP 下的价值函数

如第八章所学，MDP 下有两种价值函数：状态价值函数和动作价值函数。

$$
v_\pi(s) = \mathbb E[G_t \mid S_t=s]=\mathbb E \Big[\sum_{k=0}^T \gamma^k R_{t+k+1} \mid S_t=s\Big] \tag{10.3.1}
$$

$$
q_\pi(s,a) = \mathbb E[G_t \mid S_t=s, A_t=a]=\mathbb E \Big[\sum_{k=0}^T \gamma^k R_{t+k+1} \mid S_t=s,A_t=a\Big] \tag{10.3.2}
$$

式（10.3.1）和式（10.3.2）与 8.4 节中的关于 $v_\pi,q_\pi$ 的表达式有所不同，这里没有写出后续的含有模型信息的公式，即状态转移概率，这是因为在实际问题中，有很多场景是得不到状态转移概率的，无法做精确计算，只能使用蒙特卡洛方法来做“近似”。

