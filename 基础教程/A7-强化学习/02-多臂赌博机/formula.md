## 非平稳问题

$$
Q_{n+1}=Q_n + \alpha(R_n - Q_n), \ \ \ \alpha \in (0,1]
$$

$$
\begin{aligned}
Q_{n+1}&=Q_n + \alpha(R_n - Q_n)
\\
&=\alpha R_n + (1-\alpha)Q_n
\\
&=\alpha R_n + (1-\alpha)\alpha R_{n-1}+(1-\alpha)^2Q_{n-1}
\\
&=\alpha R_n + (1-\alpha)\alpha R_{n-1}+(1-\alpha)^2R_{n-1} + \cdots + (1-\alpha)^nQ_1
\\
&=\alpha R_n + (1-\alpha)\alpha R_{n-1}+(1-\alpha)^2R_{n-1} + \cdots + (1-\alpha)^nR_1
\end{aligned}
$$

$$
\sum_{i=1}^\infin \alpha_i = \infin \ 并且 \ \sum_{i=1}^\infin \alpha_i^2 < \infin
$$


## softmax

$$
\theta_{n+1} = \theta_{n} + \alpha \cdot \nabla J(\theta)
$$

$$
\pi_t(a) \dot= \Pr\{A_t=a\} \dot= \frac{e^{Q(a)/\tau}}{\sum_{b=1}^k e^{Q(b)/\tau}}
$$

$$
a_1=1 \to e^{a_1}=2.718
\\\\
a_2=2 \to e^{a_2}=7.389
\\\\
a_3=3 \to e^{a_3}=20.085
$$

$$
\sum_{i=1}^3 e^{a_i} = 2.718+7.389+20.085=30.192
$$

$$
P(a_1) = 2.718/30.192=0.09
\\\\
P(a_2) = 7.389/30.192=0.24
\\\\
P(a_3) = 20.085/30.192=0.67
$$

$$
Q_{t+1}(A_t) \dot= Q_t(A_t) + \alpha (R_t - \bar{R_t}) (1-\pi_t(A_t))
$$

$$
Q_{t+1}(a) \dot= Q_t(a) - \alpha (R_t - \bar{R_t}) \pi_t(a)
$$

$$

$$

$$
q_*(a) \dot= \Bbb E[R_t | A_t=a]
$$

$$
Q_t(a) \dot= \frac{执行动作a的到的收益总和}{执行动作a的次数}=\frac{\sum_{i=1}^{t-1}R_i|_{A_i=a}}{\sum_{i=1}^{t-1} 1|_{A_i=a}}
$$

$$
A_t \dot= \underset{a}{\argmax} \ Q_t(a)
$$

$$
Q_t=\frac{R_1+R_2+ \cdots \ + R_{t-1}}{t-1}=\frac{1}{t-1} \sum_{i=1}^{t-1} R_i
$$

$$
\begin{aligned}
Q_{t+1} &= \frac{1}{t} \sum_{i=1}^t R_i
\\
&=\frac{1}{t} \Big(R_t + \sum_{i=1}^{t-1} R_i \Big )
\\
&=\frac{1}{t} \Big(R_t + (t-1)\frac{1}{t-1} \sum_{i=1}^{t-1} R_i \Big )
\\
&=\frac{1}{t} \Big( R_t + (t-1) Q_t \Big )
\\
&=\frac{1}{t} \Big( R_t + t Q_t - Q_t \Big )
\\
&=Q_t + \frac{1}{t} \Big ( R_t - Q_t \Big )
\end{aligned}
$$


$$
\pi_t(a) \dot= \Pr\{A_t=a\} \dot= \frac{e^{H_t(a)}}{\sum^k_{b=1}e^{H_t(b)}}
$$

$$
H_{t+1}(A_t) \dot= H_t(A_t) + \alpha (R_t- \bar{R}_t)(1-\pi_t(A_t))
$$

$$
H_{t+1}(a) \dot= H_t(a) + \alpha (R_t- \bar{R}_t)(0-\pi_t(a))=H_t(a) - \alpha (R_t- \bar{R}_t)\pi_t(a)
$$



## UCB 算法

比如我们做一件事情 $n$ 次，然后每一次得到的reward是 $R_i$，而且做这件事情reward的期望的真值是 $R_{real}$，那么我们有

$$
P(|\frac{\sum_{i=1}^n R_i}{n} - R_{real}| \ge \epsilon ) \le e^{-\frac{-n \epsilon^2}{2 \sigma^2}}
$$

如果方差$\sigma=1，\bar{\mu}= \frac{\sum_{i=1}^n R_i}{n}, \mu=R_{real} $

$$
P(\mu \ge \bar{\mu} + \epsilon ) \le e^{-\frac{-n \epsilon^2}{2}}
$$

令 $\delta=e^{-\frac{-n \epsilon^2}{2}}$，则：$\epsilon=\sqrt{\frac{2}{n} \ln \frac{1}{\delta}}$


$$
P(\mu \ge \bar{\mu} + \sqrt{\frac{2}{n} \ln \frac{1}{\delta}} ) \le \delta
$$



这看起来很绕，其实就是：你做这件事回报的均值与期望的差距大于某个数的概率是很小的。这里给出了"大于某个数"和”这个概率应该小于多少“的具体界限。

那么这个复杂的公式对我们的老虎机决策有什么用呢？我们可以这样做：首先我们去猜一下每个赌博机的收益，然后我们去玩我们猜的”认为它收益最好“的那个。也就是说，我们给每个赌博机一个”选它的指数“，然后去选这个指数最大的那个。然后如果它不太好，我们就降低这个”选它的指数“。反之，如果我们发现它真的很好，就提高选它的指数。那么我们具体怎么猜，还有这个指数具体怎么更新呢？这就要用到刚刚那个公式了。

首先是怎么猜，我们可以每个赌博机都先来玩一次嘛，只玩一次或者很少几次，以这一次或很少的几次的值给它一个预估，来作为我们猜的依据（省的别人说我们瞎猜的对不对）。接下来，我们去更新这个指数，更新的公式为：

$$
I_i = \bar{R}_i+c\sqrt{\frac{2 \ln t}{n_i}}
$$

$$
A_t \dot= \underset{a}{\argmax} \Big[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \Big]
$$

这种基于置信度上界（upper confidence bound, UCB）的动作选择的思想是，平方根项是对第 [a] 个动作的价值估计的不确定性的度量。因此，最大值的大小是动作 [a] 的可能真实值的上限。每次选 [a] 时，不确定性可能会减小，由于 [n] 出现在不确定项的分母上，因此随着 [n] 的增加，这一项就减小了。另一方面，每次选择 [a] 之外的动作时，在分子上的 [t] 增大，而 [n] 却没有变化，所以不确定性增加了，自然对数的使用意味着随着时间的推移，增加会变得越来越小，但它是无限的。所有动作最终都将被选中，但是随着时间的流逝，具有较低价值估计的动作或者已经被选择了更多次的动作被选择的频率较低。


Chernoff-Hoeffding Bound
切诺夫界


```
step= 0,Q= [0. 0. 0. 0. 0.],               UCB= [0. 0. 0. 0. 0.],          Q+UCB= [0. 0. 0. 0. 0.],               action= 0
step= 1,Q= [-0.39  0.    0.    0.    0.  ],UCB= [0.08 0.83 0.83 0.83 0.83],Q+UCB= [-0.31  0.83  0.83  0.83  0.83],action= 1
step= 2,Q= [-0.39 -0.19  0.    0.    0.  ],UCB= [0.1  0.1  1.05 1.05 1.05],Q+UCB= [-0.28 -0.09  1.05  1.05  1.05],action= 2
step= 3,Q= [-0.39 -0.19  1.33  0.    0.  ],UCB= [0.12 0.12 0.12 1.18 1.18],Q+UCB= [-0.27 -0.08  1.44  1.18  1.18],action= 2
step= 4,Q= [-0.39 -0.19  0.18  0.    0.  ],UCB= [0.13 0.13 0.09 1.27 1.27],Q+UCB= [-0.26 -0.07  0.27  1.27  1.27],action= 3
step= 5,Q= [-0.39 -0.19  0.18  1.82  0.  ],UCB= [0.13 0.13 0.09 0.13 1.34],Q+UCB= [-0.26 -0.06  0.27  1.95  1.34],action= 3
step= 6,Q= [-0.39 -0.19  0.18  1.21  0.  ],UCB= [0.14 0.14 0.1  0.1  1.39],Q+UCB= [-0.25 -0.05  0.28  1.31  1.39],action= 4
step= 7,Q= [-0.39 -0.19  0.18  1.21  1.14],UCB= [0.14 0.14 0.1  0.1  0.14],Q+UCB= [-0.25 -0.05  0.28  1.31  1.28],action= 3
step= 8,Q= [-0.39 -0.19  0.18  1.07  1.14],UCB= [0.15 0.15 0.1  0.09 0.15],Q+UCB= [-0.24 -0.04  0.28  1.15  1.29],action= 4
step= 9,Q= [-0.39 -0.19  0.18  1.07  2.75],UCB= [0.15 0.15 0.11 0.09 0.11],Q+UCB= [-0.24 -0.04  0.28  1.15  2.85],action= 4
```



$$
P(q)=\frac{win}{win+loss}
$$


$$
P(q) = \frac{q^{\alpha-1}(1-q)^{\beta-1}}{B(\alpha, \beta)}
$$




TOPSIS

正向指标

$$
y = \frac{x-x_{min}}{x_{max}-x_{min}}
$$

逆向指标

$$
y = \frac{x_{max}-x}{x_{max}-x_{min}}
$$

计算权重值

$$
z_{ij} = \frac{y_{ij}}{\sqrt{\sum_{i=1}^n y_{ij}^2}}
$$

得到规范矩阵，n个样本，m个特征值

$$
Z = \begin{bmatrix}
z_{11} & \cdots & z_{1m}
\\
\vdots & \ddots & \vdots
\\
z_{n1} & \cdots & z_{nm}
\end{bmatrix}
$$

确定最优最劣

$$
Z^+=\big[\max (z_{11},\cdots,z_{n1}), \cdots, \max (z_{1m}, z_{nm}) \big ] = \big [Z_1^+, \cdots ,Z_m^+ \big ]
$$

$$
Z^-=\big[\min (z_{11},\cdots,z_{n1}), \cdots, \min (z_{1m}, z_{nm}) \big ] = \big [Z_1^-, \cdots ,Z_m^- \big ]
$$


$$
D_i^+=\sqrt{\sum_{j=1}^m (Z_j^+ - z_{ij})^2}
$$

$$
D_i^-=\sqrt{\sum_{j=1}^m (Z_j^- - z_{ij})^2}
$$

$$
C_i = \frac{D_i^-}{D_i^++D_i^-}
$$

$$
sort = argsort (c_i)
$$
