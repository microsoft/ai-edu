
### 拉格朗日乘子法

- 无约束优化问题

对于函数 $f(x,y)=x^2+y^2$，如果求其最小值，我们知道分别对 $x、y$ 求偏导，并令结果为 0，即可以得到结果为 $(0,0)$ 点。

- 等式约束优化问题

如果加一个限制条件，求函数 $f(x,y)$ 在约束条件 $g(x,y)=x+y+2=0$ 时的最小值。

用解方程组的方式：

$$
\begin{dcases}
    z = x^2+y^2
    \\\\
    x+y+2=0
\end{dcases}
$$
得到：
$$
z=2x^2+4x+4
$$

对$z$求$x$的导数，并令结果为 0：

$$
\frac{dz}{dx}=4x+4=0
$$

得到：当 $x=-1，y=-1$时，$z=2$ 为最小值。


- 不等式约束优化问题



### SVM

$$
\begin{aligned}
    &\min \frac{1}{2}||w||^2
    \\\\
    & s.t. \quad 1-y_i(\boldsymbol{w} \boldsymbol{x_i}+b) \le 0
\end{aligned}
\tag{19}
$$

$$
\begin{aligned}
L(w,b,\alpha)&=\frac{1}{2}||w||^2+\sum_{i=1}^n{\alpha_i}(1-y_i(\boldsymbol{w} \boldsymbol{x_i} + b))
\\\\
&=\frac{1}{2}||w||^2+\sum_{i=1}^n{(\alpha_i}-a_iy_i\boldsymbol{w} \boldsymbol{x_i} - a_iy_ib)
\end{aligned}
\tag{20}
$$

$$
\nabla_w L(w,b,a)=w - \sum a_iy_iw_i=0 \rightarrow w=\sum a_iy_ix_i \tag{21}
$$

$$
\nabla_b L(w,b,a)= -\sum a_iy_i=0 \rightarrow \sum a_iy_i=0 \tag{22}
$$

代回公式 20

$$
L(w,b,a)=-\frac{1}{2} (\sum_{i=1}^n a_iy_ix_i)^2 + \sum_{i=1}^n a_i
$$

<img src="./images/7.png" />

<center>图 7 举例</center>

$$
\begin{aligned}
    &\underset{w,b}{\min} \frac{1}{2}||w||^2
    \\\\
    & s.t. & 1-(-1)(w_1+w_2+b) \le 0
    \\\\
    &&1-(+1)(3w_1+3w_2+b) \le 0
    \\\\
    &&1-(+1)(4w_1+3w_2+b) \le 0
\end{aligned}
\tag{20}
$$

$$
\underset{\alpha}{\min} \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n[\alpha_i \alpha_j y_i y_j (x_i \cdot x_j)] - \sum_{i=1}^n\alpha_i
$$

|内积计算|$x_1$|$x_2$|$x_3$|
|--|--|--|--|
|$x_1$|$(1 \quad 1) \cdot (1 \quad 1)^T=2$|$(1 \quad 1) \cdot (3 \quad 3)^T=6$|$(1 \quad 1) \cdot (4 \quad 3)^T=7$|
|$x_2$|$(3 \quad 3) \cdot (1 \quad 1)^T=6$|$(3 \quad 3) \cdot (3 \quad 3)^T=18$|$(3 \quad 3) \cdot (4 \quad 3)^T=21$|
|$x_3$|$(4 \quad 3) \cdot (1 \quad 1)^T=7$|$(4 \quad 3) \cdot (3 \quad 3)^T=21$|$(4 \quad 3) \cdot (4 \quad 3)^T=25$|




$$
\begin{aligned}
\sum_{i=1}^n\sum_{j=1}^n[\alpha_i \alpha_j y_i y_j (x_i \cdot x_j)]&=(a_1y_1x_1+a_2y_2x_2+a_3y_3x_3)^2
\\\\
&=a_1^2y_1^2(x_1 \cdot x_1)+a_2^2y_2^2(x_2 \cdot x_2)+a_3^2y_3^2(x_3 \cdot x_3)
\\\\
&+2a_1a_2y_1y_2(x_1 \cdot x_2)+2a_1a_3y_1y_3(x_1 \cdot x_3)+2a_2a_3y_2y_3(x_2 \cdot x_3)
\\\\
&=2a_1^2+18a_2^2+25a_3^2-12a_1a_2-14a_1a_3+42a_2a_3
\end{aligned}
$$

$$
\sum_{i=1}^n\alpha_i=a_1+a_2+a_3
$$

$$

$$