## SMO（Sequential Minimal Optimization）序列最小最优化算法



在上一小节中，我们得到了对偶问题的内层最小值 $D(\alpha)$ 的表达式：

$$
D(\alpha)=\sum_{i=1}^n\alpha_i-\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha_i \alpha_j y_i y_j (x_i \cdot x_j)
\tag{5.5.8}
$$

然后就用样本实例代入来求解 $-D(\alpha)$ 的最小值，即 $D(\alpha)$ 的最大值，当然还有式 5.5.7 的约束条件。

当有 4 个或者更多样本参与计算时，式 5.5.9 会是一个 N 元二次方程，手工求解会变得非常繁琐。这一小节，我们将学习 SMO 算法，当样本量很大时，也会高效地得到解。

明确一下，目标问题是：

$$
\begin{aligned}
\underset{\alpha;\alpha \ge 0}{\min} & \quad \sum_{i=1}^n\alpha_i-\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha_i \alpha_j y_i y_j (x_i \cdot x_j)
\\\\
s.t. & \quad \sum_{i=1}^n\alpha_iy_i=0
\end{aligned}
\tag{5.5.8}
$$


