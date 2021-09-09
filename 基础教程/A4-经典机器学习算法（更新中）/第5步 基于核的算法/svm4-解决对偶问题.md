https://zhuanlan.zhihu.com/p/154517678

## SVM-4 对偶问题

### 原始问题
  
$$
\begin{aligned}
    &\underset{w,b}{\min} f(w,b)=\frac{1}{2}||w||^2
    \\\\
    & s.t. \quad 1-y_i(\boldsymbol{w} \boldsymbol{x_i}+b) \le 0 \quad \rightarrow g(x_i), i=1,...,n
\end{aligned}
\tag{15}
$$

### 构造拉格朗日函数

原始问题如公式 15 所示。按照前面的拉格朗日乘子法，先构造拉格朗日函数。但是式 15 中的约束条件与前面的例子不同，是与样本相关的，所以我们必须针对每个样本定义一个 $\alpha$ 来构造拉格朗日函数：

$$
\begin{aligned}
L(w,b,\alpha) &= f(w,b) + \alpha_1 g(x_1) + \alpha_2 g(x_2) + \cdots + \alpha_n g(x_n)
\\\\
&=f(w,b) + \sum_{i=1}^{n} \alpha_i g(x_i), \quad \alpha_i \ge 0
\end{aligned}
\tag{16}
$$

假设如果有1000个样本的话，即 n=1000，我们就会有 $\alpha_1$ 到 $\alpha_{1000}$ 个乘子。

### 极小极大问题

取 $L(w,b,\alpha)$ 关于 $\alpha$ 的最大值，并记为 $P(w,b)$：

$$
P(w,b)=\underset{\alpha}{\max} \ L(w,b,\alpha)=
\begin{cases}
    f(w,b), \qquad x \in 可行解区域内
    \\
    +\infin, \qquad x \in 可行解区域外
\end{cases}
\tag{17}
$$

- 当样本点 $x_i,y_i$ 满足 $g(x_i)$ 的要求时，$P(w,b)$ 的值等于原函数 $f(w,b)$，因为 $P(w,b)$ 的值再怎么大也不可能超过原函数，所以取值范围应该和原函数一摸一样。
- 当样本点不满足 $g(x_i)$ 的要求时，可以把乘子 $\alpha_i$ 设置为无穷的，这样$P(w,b)$ 的值可以趋近于无穷大，没有限制。

现在确定了 $\alpha$ 的值，需要解决 $w,b$ 的问题了，对 $P(w,b)$ 取关于 $w,b$ 最小值，结果记为 $p^*$：

$$
p^* = \underset{w,b}{\min} \ P(w,b)=  \underset{w,b}{\min} \ [\underset{\alpha}{\max} \ L(w,b,\alpha)] = \underset{w,b}{\min} \ f(w,b)
\tag{18}
$$

式 18 是我们要求的 SVM 问题的解，也被称为广义拉格朗日函数的**极小极大**问题。为什么要这么做呢？因为我们想用拉格朗日乘子法来解决问题。

到此为止，$p^*$ 就是原始问题式 15 的解，但是式 18 不容易求解，因为在不加入 $\alpha$ 的约束的前提下求解 $w,b$ 的最小值，这个问题不容易做，所以我们接下来用对偶问题来彻底解决。

### 对偶问题

令 $D(\alpha)$ 为 $L(w,b,\alpha)$ 关于 $w,b$ 的最小值：

$$
D(\alpha)= \underset{w,b}{\min} \ L(w,b,\alpha) \tag{19}
$$

再求 $D(\alpha)$ 关于 $\alpha$ 的最大值，并记为 $d^*$：

$$
d^*=\underset{\alpha}{\max} \ D(\alpha)= \underset{\alpha}{\max} \ [\underset{w,b}{\min} \ L(w,b,\alpha)] \tag{20}
$$

称为广义拉格朗日函数的**极大极小**问题。


### 二者关系

很直接地，如图 xx 所示，在函数 $L(w,b,\alpha)$ 内部，关于一部分参数的最小值，一定处于该函数解空间内的底部区域；相反，关于另一部分参数的最大值，一定处于该函数解空间内的顶部区域。所以有：

$$
D(\alpha) = \underset{w,b}{\min} \ L(w,b,\alpha) \le L(w,b,\alpha) \le \underset{\alpha}{\max} L(w,b,\alpha)=P(w,b) \tag{21}
$$

即：

$$
D(\alpha) \le P(x)
$$

如图 xx，在最小值区域里面的最大值，肯定要小于在最大值区域里面的最小值：

$$
\underset{\alpha}{\max}\ D(\alpha) \le \underset{w,b}{\min} \ P(w,b)
$$

即：

$$
d^*=\underset{\alpha}{\max}[\underset{w,b}{\min} \ L(w,b,\alpha)] \le \underset{w,b}{\min}[\underset{\alpha}{\max} \ L(w,b,\alpha)]=p^*
$$


### 继续求解 SVM 问题

现在求 $L$ 的最小值，对公式 20 中的 $w、b$ 求偏导（相当于对公式 6 的 $x、y$ 求偏导）：

$$
\nabla_w L(w,b,a)=w - \sum a_iy_iw_i=0 \rightarrow w=\sum a_iy_ix_i \tag{21}
$$

$$
\nabla_b L(w,b,a)= -\sum a_iy_i=0 \rightarrow \sum a_iy_i=0 \tag{22}
$$

请注意，此时的 $x_i、y_i$ 不是变量，是样本值，可以看成是固定变量。

公式 21、22 代回公式 20：

$$
L_{min(w,b)} = \underset{w,b}{\min} L(w,b,a)=-\frac{1}{2} (\sum_{i=1}^n a_iy_ix_i)^2 + \sum_{i=1}^n a_i \tag{23}
$$

接下来求公式 23 的极大值 $\underset{a}{max} L_{min(w,b)}$，两边都乘以-1，就把求极大值问题变为求极小值问题：

$$
L_{max(a)}=\underset{a}{\max} L_{min(w,b)} = - \underset{a}{\min} L_{min(w,b)} =\sum_{i=1}^n a_i-\frac{1}{2} (\sum_{i=1}^n a_iy_ix_i)^2 \tag{24}
$$


再对式 24 中的 $\alpha_i$ 求偏导，即可求得 $\alpha_i$ 的值。







对式 24 分别求 $\alpha_2、\alpha_3$ 的偏导，并令结果等于 0：

$$
\begin{cases}
\nabla_{\alpha_2} L_{max(a)}=8\alpha_2+10\alpha_3-2=0
\\\\
\nabla_{\alpha_3} L_{max(a)}=13\alpha_3+10\alpha_2-2=0
\end{cases}
\tag{25}
$$

解得：

$$
\begin{cases}
    \alpha_1=0.5
    \\\\
    \alpha_2=1.5
    \\\\
    \alpha_3=-1
\end{cases}
$$

其中，$\alpha_3=-1$ 违反了公式 8 的约定。我们看一下公式 24 的形态，运行 test.py 以得到图 10，同时输出打印信息如下：

```
左：a2=1.50, a3=-1.00, z=-0.50
右：a2=0.25, a3=0.00, z=-0.25
```

<img src="./images/10.png" />

<center>图 10 </center>



图 10 的左子图，为 $\alpha_2、\alpha_3$ 在自由取值空间内的函数形态，极小值确实在 $(\alpha_2=1.5,\alpha_3=-1,z=-0.5)$ 上。由于公式 8 的约定，我们把 $\alpha_2、\alpha_3$ 的取值限制在 $(0,+\infin)$上，得到右子图的形态，可以看到极值点在$(\alpha_2=0.25,\alpha_3=0,z=-0.25)$ 上。

我们也可以分别令 $\alpha_2=0$ 和 $\alpha_3=0$，来得到公式 24 的解：

令 $\alpha_2=0$，则 $L_{max(a)}=6.5\alpha_3^2-2\alpha_3，\nabla_{\alpha_3} L_{max(a)}=13\alpha_3-2=0，\alpha_3=\frac{2}{13}，L_{max(a)}=-\frac{2}{13}$。

令 $\alpha_3=0$，则 $L_{max(a)}=4\alpha_2^2-2\alpha_2，\nabla_{\alpha_2} L_{max(a)}=8\alpha_2-2=0，\alpha_2=\frac{1}{4}，L_{max(a)}=-\frac{1}{4}$。

由于 $-\frac{1}{4} < -\frac{2}{13}$，为极小值，所以我们最终得到：

$\alpha_2=\frac{1}{4}，\alpha_3=0，\alpha_1=\alpha_2+\alpha_3=\frac{1}{4}$，极值为$-\frac{1}{4}$。

继续解出$w$和$b$。


### 思考与练习

