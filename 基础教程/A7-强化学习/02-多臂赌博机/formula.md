
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
A_t \dot= \underset{a}{\argmax} \Big[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \Big]
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