
$$
p(s'|s,a) = \Pr \{S_t=s'|S_{t-1}=s,A_{t-1}=a\}
$$

$$
\sum_{i=0}^n p(s_i'|s_j,a) = 1
$$

$$
p(s_1'|s_j,a) + p(s_2'|s_j,a) + p(s_3'|s_j,a)  = 1
$$

$$
P = 
\begin{bmatrix}
p(s_1|s_1) & p(s_2|s_1) & \cdots & p(s_n|s_1)
\\
p(s_1|s_2) & p(s_2|s_2) & \cdots & p(s_n|s_2)
\\
\vdots & \vdots & \ddots & \vdots
\\
p(s_1|s_n) & p(s_2|s_n) & \cdots & p(s_n|s_n)
\end{bmatrix}
$$

奖励函数

$$
R(s)=\mathbb {E} \ [R_{t} \ | \ S_{t-1}=s,A_{t-1}=a ]
$$

$$
R(s)=\mathbb {E} \ [R_{t} \ | \ S_{t-1}=s ]
$$


$$
\begin{aligned}
G_t &= R_{t+1} + \gamma R_{t+2}  + \gamma^2 R_{t+3} + \cdots +  \gamma^{T-t-1} R_{T}
\\
&= \sum_{k=0}^{T} \gamma^k R_{t+k+1}, \ 0 \le \gamma \le 1
\end{aligned}
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
R_s = \mathbb{E} [R_{t+1} | S_t=s]
$$

$$
\begin{aligned}
V(s) &= \mathbb{E} [G_t \ | \ S_t=s]
\\
&=\mathbb{E} [R_{t+1} + \gamma R_{t+2}  + \gamma^2 R_{t+3} + \cdots \ | \ S_t=s]
\\
&=\mathbb{E} [R_{t+1} + \gamma G_{t+1} \ | \ S_t=s]
\\
&=\mathbb{E} [R_{t+1} + \gamma V(s_{t+1}) \ | \ S_t=s]
\\
&=\mathbb{E} [R_{t+1}] + \gamma\mathbb{E}[V(s_{t+1})]
\\
&=R + \gamma 
\end{aligned}
$$

$$
V(Class3) = 4.09
\\
\begin{aligned}
X&=R_{Class3}+\gamma*[V(Pub)*P(S_{Class3}|S_{Pub}) 
\\
&+ V(Pass)*P(S_{Class3}|S_{Pass})]
\\
&=(-2)+0.9*(1.93*0.4+10*0.6)=4.09
\end{aligned}
\\
V(Class3) == X
$$

$$
V(s) = R_s + \gamma * \sum V(s') P(s|s')
$$

$$
V(s)=R_s + \gamma \sum_{s' \in S} Pss' \cdot V(s')
$$

$$
V(s)=R_s + \gamma * [p_1V(s'_1) + p_2V(s'_2) + p_3V(s'_3)]
$$