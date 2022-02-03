
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
p(s_1|s_1) & p(s_1|s_2) & \cdots & p(s_1|s_n)
\\
p(s_2|s_1) & p(s_2|s_2) & \cdots & p(s_2|s_n)
\\
\vdots & \vdots & \ddots & \vdots
\\
p(s_n|s_1) & p(s_n|s_2) & \cdots & p(s_n|s_n)
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
&=\mathbb{E} [R_{t+1}] + \gamma \mathbb{E} [G_{t+1}|S_t=s]
\\
&=R_{t+1} + \gamma V(s_{t+1}) 
\end{aligned}
$$

$$
V(Class3) = 4.09
\\
\begin{aligned}
X&=R_{Class3}+\gamma*[V(Pub)*P(S_{Class3}|S_{Pub}) 
\\
&+ V(A_{Pass})*P(S_{Class3}|S_{A_{Pass}})]
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

矩阵形式

$$
V = R + \gamma PV
$$

$$
\begin{bmatrix}
V(1)
\\
V(2)
\\
\vdots
\\
V(n)
\end{bmatrix}
=\
\begin{bmatrix}
R_1
\\
R_2
\\
\vdots
\\
R_n
\end{bmatrix}
+\gamma
\begin{bmatrix}
P_{11} & P_{12} & \cdots & P_{1n}
\\
P_{21} & P_{22} & \cdots & P_{2n}
\\
\vdots & \vdots & \ddots & \vdots
\\
P_{n1} & P_{n2} & \cdots & P_{nn}
\end{bmatrix}
\begin{bmatrix}
V(1)
\\
V(2)
\\
\vdots
\\
V(n)
\end{bmatrix}
$$

$$
V - \gamma PV = R
\\
(1-\gamma {})V = R
\\
V = (I - \gamma P)^{-1} R
$$

策略价值函数

$$
v_{\pi}(s)=\mathbb {E}_{\pi} [ G_t |S_t=s]
$$

$$
\begin{aligned}
v_{\pi}(s)&=\sum_{a \in A} \pi(a|s) q_\pi(s,a)
\\
&=\pi(a_1|s) q_{\pi}(s,a_1)+\pi(a_2|s) q_{\pi}(s,a_2)+\pi(a_3|s) q_{\pi}(s,a_3)
\end{aligned}
$$

策略动作函数

$$
q_{\pi}(s,a)=\mathbb E_{\pi} [G_t | S_t=s, A_t=a]
$$


$$
\begin{aligned}
q_{\pi}(s,a)&=R_s^a + \gamma \sum_{s' \in S} P^a_{ss'} v_{\pi}(s')
\\
&= R_s^a + \gamma [P_1 v_{\pi}(s'_1)+P_2 v_{\pi}(s'_2)]
\end{aligned}
$$

$$
v_{\pi}(s)=\sum_{a \in A} \pi(a|s)\Big[ R_s^a + \gamma \sum_{s' \in S} P^a_{ss'} v_{\pi}(s') \Big]
$$

$$
q_{\pi}(s,a)=R_s^a + \gamma \sum_{s' \in S} P^a_{ss'} \sum_{a' \in A} \pi(a'|s') q_\pi(s',a')
$$

$$
\begin{aligned}
V_1 &= \pi(A_{Play}|S_{V_1})*(R_{Play}+\gamma P_{11}V_1)+\pi(A_{Quit}|S_{V_2})*(R_{Quit}+\gamma P_{12}V_2)
\\
V_2 &= \pi(A_{Play}|S_{V_2})*(R_{Play}+\gamma P_{21}V_1)+\pi(A_{Study1}|S_{V_2})*(R_{Study1}+\gamma P_{23}V_3)
\\
V_3 &= \pi(Sleep|S_{V_3})*(R_{Sleep}+\gamma P_{30}V_0)+\pi(A_{Study2}|S_{V_3})*(R_{Study2}+\gamma P_{34}V_4)
\\
V_4 &= \pi(A_{Pass}|S_{V_4})*(R_{Pass}+\gamma P_{40}V_0)+\pi(A_{Pub}|S_{V_4})*(R_{Pub}+\gamma P_{42}V_2+\gamma P_{43}V_3+\gamma P_{44}V_4)
\end{aligned}
$$