
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
RMSE=\sqrt {\frac{1}{n} \sum_{i=1}^n (a_i-y_i)^2}
$$

比如 

$$
y=[0,1,2], a=[0.1,1,1.8], b=[0.1,1.1,2.2]
$$

$$
MSE(a,y)=[(0.1-0)^2 + (1-1)^2 + (1.8-2)^2)]/3=0.0167
\\
MSE(b,y)=[(0.1-0)^2 + (1.1-1)^2 + (2.2-2)^2)]/3=0.02
$$

$$
RMSE(a,y)=\sqrt{[(0.1-0)^2 + (1-1)^2 + (1.8-2)^2)]/3}=0.129
\\
RMSE(b,y)=\sqrt{[(0.1-0)^2 + (1.1-1)^2 + (1.8-2)^2)]/3}=0.141
$$

$$
\begin{aligned}
G_1 &= R_1
\\
G_2 &= G_1 + \gamma R_2=R_1+\gamma R_2
\\
G_3 &= G_2 + \gamma^2 R_3 = R_1 + \gamma R_2 + \gamma^2 R_3
\\
G_4 &= G_3 + \gamma^3 R_4 = R_1 + \gamma R_2 + \gamma^2 R_3 + \gamma^3 R_4
\end{aligned}
$$


$$
\begin{aligned}
G[4] &= R_4
\\
G[3] &= \gamma G[4]+R_3 = R_3 + \gamma R_4
\\
G[2] &= \gamma G[3]+R_2 = R_2 + \gamma R_3 + \gamma^2 R_4
\\
G[1] &= \gamma G[2]+R_1 = R_1 + \gamma R_2 + \gamma^2 R_3 + \gamma^3 R_4 
\end{aligned}
$$

$$
\begin{aligned}
G_1 &= R_4, & V[S_{9}]+=G_1
\\
G_2 &= \gamma G_1 + R_3 = R_3 + \gamma R_4, & V[S_5]+=G_2
\\
G_3 &= \gamma G_2 + R_2 = R_2 + \gamma R_3 + \gamma^2 R_4, & V[S_4]+=G_3
\\
G_4 &= \gamma G_3 + R_1 = R_1 + \gamma R_2 + \gamma^2 R_3 + \gamma^3 R_4 , & V[S_0] += G_4
\end{aligned}
$$

$$
V(S_t) = V(S_t) + \alpha [G_{batch} - V(S_t)]
$$

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]
$$

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1},a) - Q(S_t,A_t)]
$$


$$
V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1})-V(S_t)]
$$

$$
\begin{aligned}
Q(S_t,A_t) & \leftarrow Q(S_t,A_t) + \alpha [R_{t+1} + \gamma \mathbb{E}[Q(S_{t+1},A_{t+1})|S_{t+1}] - Q(S_t,A_t)]
\\
& \leftarrow Q(S_t,A_t) + \alpha [R_{t+1} + \gamma \sum_a \pi (a|S_{t+1}) Q(S_{t+1},a) - Q(S_t,A_t)]
\end{aligned}
$$