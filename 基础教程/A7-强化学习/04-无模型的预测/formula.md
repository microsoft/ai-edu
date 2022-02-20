
$$
input \ x_1
\\
input \ x_2
\\ 
avg = (x_1 + x_2)/2
\\
input \ x_3
\\
avg = (avg * 2 + x_3)/3
\\
\cdots
\\
input \ x_n
\\
avg = [avg * (n-1) + x_n]/n = avg + \frac{1}{n}(x_n - avg)
$$

$$
N(s_t) = N(s_t) + 1
\\
V(s_t) = V(s_t) + \frac{1}{N(s_t)}(G_t - V(s_t))
\\
V(s_t) = V(s_t) + \alpha \big [G_t - V(s_t) \big ]
$$

$$
\begin{aligned}
G_t &= R_{t+1} + \gamma R_{t+2}  + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \cdots
\\
&= R_{t+1} + \gamma (R_{t+2}  + \gamma R_{t+3} + \gamma^{2} R_{t+4}+\cdots)
\\
&=R_{t+1} + \gamma G_{t+1}
\end{aligned}
$$

$$
\begin{aligned}
G_t = R_{t+1} + \gamma G_{t+1}
\end{aligned}
$$


$$
\begin{aligned}
V(s_t) &= \mathbb{E} [G_t]
\\
V(s_{t+1}) &= \mathbb{E} [G_{t+1}]
\end{aligned}
$$

$$
V(s_t) = V(s_t) + \alpha \big[ R_{t+1} + \gamma V(s_{t+1}) - V(s_t) \big]
$$


均方差
$$
MSE = \frac{1}{n} \sum_{i=1}^n (a_i-y_i)^2
$$
a：预测值
y：标准值

均方根误差
$$
RSME=\sqrt {\frac{1}{n} \sum_{i=1}^n (a_i-y_i)^2}
$$