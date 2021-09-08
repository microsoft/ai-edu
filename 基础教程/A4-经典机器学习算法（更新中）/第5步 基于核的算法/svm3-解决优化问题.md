https://zhuanlan.zhihu.com/p/154517678




### SVM-3

针对 SVM 的优化问题：

$$
\begin{aligned}
    &\min \ f(w,b)=\frac{1}{2}||w||^2
    \\\\
    & s.t. \quad 1-y_i(\boldsymbol{w} \boldsymbol{x_i}+b) \le 0
\end{aligned}
\tag{19}
$$

我们使用拉格朗日乘子法解决。约束条件中有很多样本，而不是单一变量，此时可以对每个样本都看作是一个独立的不等式约束条件，所以需要在每个样本上都附加一个$\alpha_i$，则构造出的拉格朗日公式的原型为：

$$
L = f(x) + \alpha_1 g(x_1) + \alpha_2 g(x_2) + \cdots + \alpha_n g(x_n)=f(x) + \sum_{i=1}^{n} \alpha_i g(x_i)
$$

应用于式 19 上：

$$
\begin{aligned}
L(w,b,\alpha)&=\frac{1}{2}||\boldsymbol{w}||^2+\sum_{i=1}^n{\alpha_i}(1-y_i(\boldsymbol{w} \boldsymbol{x_i} + b))
\\\\
&=\frac{1}{2}||w||^2+\sum_{i=1}^n{\alpha_i}-\boldsymbol{w}\sum_{i=1}^n\alpha_iy_i \boldsymbol{x_i} - b\sum_{i=1}^n\alpha_iy_i
\end{aligned}
\tag{20}
$$

求 $w,b$ 的偏导

$$
\nabla_w L=\boldsymbol{w}-\sum_{i=1}^n\alpha_iy_ix_i=0 \rightarrow \boldsymbol{w}=\sum_{i=1}^n\alpha_iy_ix_i \tag{21}
$$
$$
\nabla_bL=-\sum_{i=1}^n\alpha_iy_i=0 \rightarrow \sum_{i=1}^n\alpha_iy_i=0 \tag{22}
$$

将式 21、22 代回式 20，把第 1、3 项的 $\boldsymbol{w}$ 替换，消掉第 4 项

$$
\begin{aligned}
L(w,b,\alpha)&=\frac{1}{2} \left(\sum_{i=1}^n\alpha_iy_ix_i \right)^2+\sum_{i=1}^n\alpha_i- \left(\sum_{i=1}^n\alpha_iy_ix_i \right) \left(\sum_{i=1}^n\alpha_iy_ix_i \right)
\\\\
&=\sum_{i=1}^n\alpha_i-\frac{1}{2}(\sum_{i=1}^n\alpha_iy_ix_i)^2
\end{aligned}
\tag{23}
$$

然后再解公式 23，该式中已经没有 $w,b$ 了，只需要面对 $\alpha_i$。注意 $x_i,y_i$ 是样本数据，不是变量。

下面我们举例说明，见图 7。

<img src="./images/7.png" />

<center>图 7 举例</center>

样本

|序号|$x_1$|$x_2$|标签 $y$|
|--|--|--|--|
|$p_1$|1|1|-1|
|$p_2$|3|3|+1|
|$p_3$|4|3|+1|

优化目标为：

$$
\begin{aligned}
    &\underset{w,b}{\min} f(w,b)=\frac{1}{2}||w||^2
    \\\\
    & s.t. \quad 1-(-1)(w_1+w_2+b) \le 0
    \\\\
    & \qquad \quad 1-(+1)(3w_1+3w_2+b) \le 0
    \\\\
    & \qquad \quad 1-(+1)(4w_1+3w_2+b) \le 0

\end{aligned}
\tag{20}
$$


|内积计算|$x_1$|$x_2$|$x_3$|
|--|--|--|--|
|$x_1$|$(1 \ 1) \cdot (1 \ 1)^T=2$|$(1 \ 1) \cdot (3 \ 3)^T=6$|$(1 \ 1) \cdot (4 \ 3)^T=7$|
|$x_2$|$(3 \ 3) \cdot (1 \ 1)^T=6$|$(3 \ 3) \cdot (3 \ 3)^T=18$|$(3 \ 3) \cdot (4 \ 3)^T=21$|
|$x_3$|$(4 \ 3) \cdot (1 \ 1)^T=7$|$(4 \ 3) \cdot (3 \ 3)^T=21$|$(4 \ 3) \cdot (4 \ 3)^T=25$|


$$
\begin{aligned}
L(w,b,\alpha)&=\sum_{i=1}^n \alpha_i-\frac{1}{2} (\sum_{i=1}^n \alpha_iy_ix_i)^2 
\\\\
&=(\alpha_1+\alpha_2+\alpha_3)-\frac{1}{2}(\alpha_1y_1x_1+\alpha_2y_2x_2+\alpha_3y_3x_3)^2 \quad(带入y_i)
\\\\ 
&=(\alpha_1+\alpha_2+\alpha_3)-\frac{1}{2}(-\alpha_1x_1+\alpha_2x_2+\alpha_3x_3)^2 \quad(展开平方项)
\\\\
&=(\alpha_1+\alpha_2+\alpha_3)-\frac{1}{2} [ \alpha_1^2(x_1 \cdot x_1)+\alpha_2^2(x_2 \cdot x_2)+\alpha_3^2(x_3 \cdot x_3)
\\\\
& \quad -2\alpha_1\alpha_2(x_1 \cdot x_2)-2\alpha_1\alpha_3(x_1 \cdot x_3)+2\alpha_2\alpha_3(x_2 \cdot x_3) ]
\\\\
&=(\alpha_1+\alpha_2+\alpha_3)-\frac{1}{2}(2\alpha_1^2+18\alpha_2^2+25\alpha_3^2-12\alpha_1\alpha_2-14\alpha_1\alpha_3+42\alpha_2\alpha_3)
\end{aligned}
\tag{23}
$$

还有一个附加条件，由公式 22 得知：

$$
\sum_{i=1}^n \alpha_iy_i=\alpha_1 \cdot (-1) + \alpha_2 \cdot 1 + \alpha_3 \cdot  1=0，即：\alpha_1=\alpha_2+\alpha_3 \tag{24}
$$

带入式 23：

$$
L(w,b,\alpha)=4\alpha_2^2+6.5\alpha_3^2+10\alpha_2\alpha_3-2\alpha_2-2\alpha_3 \tag{25}
$$

式 25 中的 $\alpha_2,\alpha_3$ 如何求解呢？需要用到拉格朗日对偶问题。
