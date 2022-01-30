


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
v(s) &= \mathbb{E} [G_t \ | \ S_t=s]
\\
&=\mathbb{E} [R_{t+1} + \gamma R_{t+2}  + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \cdots \ | \ S_t=s]
\\
&=\mathbb{E} [R_{t+1} + \gamma G_{t+1} \ | \ S_t=s]
\\
&=\mathbb{E} [R_{t+1} + \gamma v(s_{t+1}) \ | \ S_t=s]
\end{aligned}
$$