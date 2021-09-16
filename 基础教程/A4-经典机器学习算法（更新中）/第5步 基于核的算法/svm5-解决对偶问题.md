https://zhuanlan.zhihu.com/p/154517678

##  解决对偶问题

### 原始问题


$$
\begin{aligned}
    &\min \ f(w,b)=\frac{1}{2}||\boldsymbol{w}||^2
    \\\\
    & s.t. \quad 1-y_i(\boldsymbol{w} \boldsymbol{x_i}+b) \le 0, \quad i=1,...,n
\end{aligned}
\tag{5.3.1}
$$


### 拉格朗日函数

$$
\begin{aligned}
L(w,b,\alpha)&=\frac{1}{2}||\boldsymbol{w}||^2+\sum_{i=1}^n{\alpha_i}(1-y_i(\boldsymbol{w} \boldsymbol{x_i} + b))
\\\\
&=\frac{1}{2}||\boldsymbol{w}||^2+\sum_{i=1}^n{\alpha_i}-\boldsymbol{w}\sum_{i=1}^n\alpha_iy_i \boldsymbol{x_i} - b\sum_{i=1}^n\alpha_iy_i
\end{aligned}
\tag{5.3.3}
$$

### 求极小值

$$
D(\alpha)= \underset{w,b}{\min} \ L(w,b,\alpha) \tag{5.4.8}
$$

按照拉格朗日函数的一般解法，求原始变量对函数的偏导数，并令其结果为 0。在本例中，原始变量是 $w,b$，所以我们求 $L$ 对 $w,b$ 的偏导：

$$
\nabla_w L=\boldsymbol{w}-\sum_{i=1}^n\alpha_iy_ix_i=0 \rightarrow \boldsymbol{w}=\sum_{i=1}^n\alpha_iy_ix_i \tag{5.3.4}
$$
$$
\nabla_bL=-\sum_{i=1}^n\alpha_iy_i=0 \rightarrow \sum_{i=1}^n\alpha_iy_i=0 \tag{5.3.5}
$$

将式 5.3.4、5.3.5 代回式 5.3.3，把第 1、3 项的 $\boldsymbol{w}$ 替换，消掉第 4 项（值为 0），即得到 $L$ 关于 $w,b$ 的最小值表达式：

$$
\begin{aligned}
D(\alpha) = \underset{w,b}{\min} \ L(w,b,\alpha)&=\frac{1}{2} \left(\sum_{i=1}^n\alpha_iy_ix_i \right)^2+\sum_{i=1}^n\alpha_i- \left(\sum_{i=1}^n\alpha_iy_ix_i \right) \left(\sum_{i=1}^n\alpha_iy_ix_i \right)
\\\\
&=\sum_{i=1}^n\alpha_i-\frac{1}{2}(\sum_{i=1}^n\alpha_iy_ix_i)^2
\end{aligned}
\tag{5.3.6}
$$

到这里，式 5.3.6 中已经没有 $w,b$ 了，注意 $x_i,y_i$ 是样本数据，不是变量，所以只需要面对 $\alpha_i$。

### 实例推导

我们把样本实例表 5.3.1 中的数据代入式 5.3.6：

$$
\begin{aligned}
D(\alpha) &=\sum_{i=1}^n \alpha_i-\frac{1}{2} (\sum_{i=1}^n \alpha_iy_ix_i)^2 
\\\\
&=(\alpha_1+\alpha_2+\alpha_3)-\frac{1}{2}(\alpha_1y_1x_1+\alpha_2y_2x_2+\alpha_3y_3x_3)^2 \quad(带入y_i的值)
\\\\ 
&=(\alpha_1+\alpha_2+\alpha_3)-\frac{1}{2}(-\alpha_1x_1+\alpha_2x_2+\alpha_3x_3)^2 \quad(展开平方项)
\\\\
&=(\alpha_1+\alpha_2+\alpha_3)-\frac{1}{2} [ \alpha_1^2(x_1 \cdot x_1)+\alpha_2^2(x_2 \cdot x_2)+\alpha_3^2(x_3 \cdot x_3)
\\\\
& \quad -2\alpha_1\alpha_2(x_1 \cdot x_2)-2\alpha_1\alpha_3(x_1 \cdot x_3)+2\alpha_2\alpha_3(x_2 \cdot x_3) ] \quad(代入x的内积计算结果)
\\\\
&=(\alpha_1+\alpha_2+\alpha_3)-\frac{1}{2}(2\alpha_1^2+18\alpha_2^2+25\alpha_3^2-12\alpha_1\alpha_2-14\alpha_1\alpha_3+42\alpha_2\alpha_3)
\end{aligned}
\tag{5.3.7}
$$

其中，($x_i \cdot x_j$) 的内积计算结果如表 5.3.2 所示。

表 5.3.2 内积计算结果

|内积计算|$x_1$|$x_2$|$x_3$|
|--|--|--|--|
|$x_1$|$(1 \ 1) \cdot (1 \ 1)^T=2$|$(1 \ 1) \cdot (3 \ 3)^T=6$|$(1 \ 1) \cdot (4 \ 3)^T=7$|
|$x_2$|$(3 \ 3) \cdot (1 \ 1)^T=6$|$(3 \ 3) \cdot (3 \ 3)^T=18$|$(3 \ 3) \cdot (4 \ 3)^T=21$|
|$x_3$|$(4 \ 3) \cdot (1 \ 1)^T=7$|$(4 \ 3) \cdot (3 \ 3)^T=21$|$(4 \ 3) \cdot (4 \ 3)^T=25$|


还有一个附加条件，由式 5.3.5 得知：

$$
\sum_{i=1}^n \alpha_iy_i=\alpha_1 \cdot (-1) + \alpha_2 \cdot 1 + \alpha_3 \cdot 1=0，即：\alpha_1=\alpha_2+\alpha_3 \tag{5.3.8}
$$

带入式 5.3.7，消掉 $\alpha_1$：

$$
D(\alpha)=2\alpha_2 + 2\alpha_3 -(4\alpha_2^2+6.5\alpha_3^2+10\alpha_2\alpha_3) \tag{5.3.9}
$$

得到式 5.3.9 后，我们暂时失去了求解目标，因为不知道下面要做什么了。 $\alpha_2,\alpha_3$ 如何求解呢？这就需要用到拉格朗日对偶问题了，我们在下一小节中学习。


### 求极大值



接下来求公式 5.4.6 的极大值 $d^*=\underset{\alpha}{\max} D(\alpha)$。求 $D(\alpha)$ 这个凹函数的极大值，等价于求 $-D(\alpha)$ 这个凸函数的极小值，所以对 $D(\alpha)$ 取负号得到：

$$
-D(\alpha)=4\alpha_2^2+6.5\alpha_3^2+10\alpha_2\alpha_3-2\alpha_2 - 2\alpha_3 \tag{5.4.12}
$$

求极小值的方法，我们前面实践过很多次了：对表达式式中的变量求偏导，再令结果为 0 即可。

对式 5.4.12 分别求 $\alpha_2、\alpha_3$ 的偏导，并令结果等于 0：

$$
\begin{cases}
\nabla_{\alpha_2} [-D(\alpha)] = 8\alpha_2+10\alpha_3-2=0
\\\\
\nabla_{\alpha_3} [-D(\alpha)]=13\alpha_3+10\alpha_2-2=0
\end{cases}
\tag{5.4.13}
$$

解得：

$$
\begin{cases}
    \alpha_1=\alpha_2 + \alpha_3=0.5 \quad (式 5.3.8)
    \\\\
    \alpha_2=1.5
    \\\\
    \alpha_3=-1
\end{cases}
$$

其中，$\alpha_3=-1$ 违反了公式 5.4.2 关于 $\alpha_i \ge 0$ 的约定，这是为什么呢？因为从图 5.3.1 来看，样本 $p_3$ 不是关键点（支持向量），它不参与计算，所以 $\alpha_3$ 的值应该为 0。

我们看一下公式 5.4.12 在三维空间中的形态来具体理解。运行 4-Lmin.py 以得到图 5.4.2，同时输出打印信息如下：

```
左：a2=1.50, a3=-1.00, z=-0.50
右：a2=0.25, a3=0.00, z=-0.25
```

<img src="./images/5.4.2.png" />
<center>图 5.4.2 </center>

图 5.4.2 的左子图，为 $\alpha_2、\alpha_3$ 在自由取值空间内（用 [-2,2] 近似表示）的函数形态，极小值确实在 $(\alpha_2$=1.5, $\alpha_3$=-1, $z$=-0.5) 上。

由于公式 5.4.2 的约定，我们把 $\alpha_2、\alpha_3$ 的取值限制在 $(0,+\infin)$上（用 [0,0.3] 近似表示），得到右子图的形态，右子图是左子图的一部分。可以看到极值点在 $(\alpha_2$=0.25, $\alpha_3$=0, $z$=-0.25) 上。

我们也可以分别令 $\alpha_2=0$ 和 $\alpha_3=0$，来得到公式 5.4.12 的解：

- 令 $\alpha_2=0$
  
  则 $-D(\alpha)=6.5\alpha_3^2-2\alpha_3$，求导：$13\alpha_3-2=0，\alpha_3=\frac{2}{13}，d^*_1=-\frac{2}{13}$。

  注意，这里不要错误地令 $-D(\alpha)=6.5\alpha_3^2-2\alpha_3=0$ 来求一元二次方程的解，因为我们是求函数极值，而不是求方程的解。

- 令 $\alpha_3=0$

    则 $-D(\alpha)=4\alpha_2^2-2\alpha_2，求导： 8\alpha_2-2=0，\alpha_2=\frac{1}{4}，d^*_2=-\frac{1}{4}$。

由于 $-\frac{1}{4} < -\frac{2}{13}$，为极小值，所以我们最终得到：

$\alpha_2=\frac{1}{4}，\alpha_3=0，\alpha_1=\alpha_2+\alpha_3=\frac{1}{4}$，$d^*=-\frac{1}{4}$。


### 思考与练习

