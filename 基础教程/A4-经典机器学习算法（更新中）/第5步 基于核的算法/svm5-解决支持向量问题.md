
## 解决分类间隔和支持向量问题


### 继续求解 SVM 问题

有了式 5.4.6，解题思路就清晰了。在 5.3 小节中，我们已经通过求 $w,b$ 的偏导，顺利地得到了 $D(\alpha)=\underset{w,b}{\min} L(w,b,a)$ 的部分，得到式 5.3.9。按照式 5.4.5 的定义，重新命名为：

$$
D(\alpha)=2\alpha_2 + 2\alpha_3 -(4\alpha_2^2+6.5\alpha_3^2+10\alpha_2\alpha_3) \tag{5.4.11}
$$

接下来求公式 5.4.6 的极大值 $d^*=\underset{\alpha}{\max} D(\alpha)$。求 $D(\alpha)$ 这个凹函数的极大值，等价于求 $-D(\alpha)$ 这个凸函数的极小值，所以对 $D(\alpha)$ 取负号得到：

$$
L_{min(\alpha)} = \underset{\alpha}{\max} \ D(\alpha) =\underset{\alpha}{\min} \ [-D(\alpha)]=4\alpha_2^2+6.5\alpha_3^2+10\alpha_2\alpha_3-2\alpha_2 - 2\alpha_3 \tag{5.4.12}
$$

求极小值的方法，我们前面实践过很多次了：对表达式式中的变量求偏导，再令结果为 0 即可。对式 5.4.12 分别求 $\alpha_2、\alpha_3$ 的偏导，并令结果等于 0：

$$
\begin{cases}
\nabla_{\alpha_2} L_{min(\alpha)}=8\alpha_2+10\alpha_3-2=0
\\\\
\nabla_{\alpha_3} L_{min(\alpha)}=13\alpha_3+10\alpha_2-2=0
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

其中，$\alpha_3=-1$ 违反了公式 5.4.2 关于 $\alpha_i \ge 0$ 的约定，这是为什么呢？因为从图 5.3.1 来看，样本 $p_3$ 好像不是关键点（支持向量），它不参与计算，所以 $\alpha_3$ 的值应该为 0。

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
  
  则 $L_{min(\alpha)}=6.5\alpha_3^2-2\alpha_3$，求导：$\nabla_{\alpha_3} L_{min(\alpha)}=13\alpha_3-2=0，\alpha_3=\frac{2}{13}，L_{min(\alpha)}=-\frac{2}{13}$。

  注意，这里不要错误地令 $L_{min(\alpha)}=6.5\alpha_3^2-2\alpha_3=0$ 来求一元二次方程的解，因为我们是求函数极值，而不是求方程的解。

- 令 $\alpha_3=0$

    则 $L_{min(\alpha)}=4\alpha_2^2-2\alpha_2，\nabla_{\alpha_2} L_{min(\alpha)}=8\alpha_2-2=0，\alpha_2=\frac{1}{4}，L_{min(\alpha)}=-\frac{1}{4}$。

由于 $-\frac{1}{4} < -\frac{2}{13}$，为极小值，所以我们最终得到：

$\alpha_2=\frac{1}{4}，\alpha_3=0，\alpha_1=\alpha_2+\alpha_3=\frac{1}{4}$，极值为$-\frac{1}{4}$。



### 为什么有些 $\alpha$ 的值为 0 ?

得到 $\alpha_i$ 值后，还没有结束，我们的最终目标是得到分界线和分类间隔的表达式，即求得 $w$ 和 $b$。但是在那之前，我们先看看为什么 $a_3=0$ ？

注意到拉格朗日函数中 $\alpha_i$ 的定义是针对样本点的，所以我们把样本点都绘制在图 5.5.1（与图 5.3.1 一样）中，可以看到最终决定分类间隔的样本点是 $p_1,p_2$，而$p_3$ 在分类间隔以外，不参与决策，因此 $a_1$ 和 $a_2$ 都有值，而 $a_3=0$。

<img src="./images/5.3.1.png" />
<center>图 5.5.1 样本及其分类间隔</center>

### 求解 $\boldsymbol{w}$ 值

接下来求 $\boldsymbol{w}$ 的值。

在 5.3 小节中，式 5.3.4 已经给了我们答案：

$$
\boldsymbol{w}=\sum_{i=1}^n\alpha_iy_i \boldsymbol{x}_i \tag{5.5.1}
$$

注意，由于 $\boldsymbol{x}_i$ 是一个向量，所以最后的 $\boldsymbol{w}$ 也是个向量，在本例中是一个二维的向量。

表 5.5.1 计算 $\boldsymbol{w}$ 所需要的各个因子

|样本 $i$|$x_{i,1}$|$x_{i,2}$|标签 $y_i$|乘子 $a_i$
|--|--|--|--|--|
|$x_1$|1|1|-1|0.25|
|$x_2$|3|3|+1|0.25|
|$x_3$|4|3|+1|0|

表 5.5.1 给出了式 5.5.1 所需要的所有计算因子，具体计算如下：

$$
\begin{aligned}
\boldsymbol{w} &= a_1 y_1 \boldsymbol{x}_1 + a_2 y_2 \boldsymbol{x}_2 + a_3 y_3 \boldsymbol{x}_3
\\\\
&=0.25*(-1)*(1 \ 1)+0.25*1*(3 \ 3)+0*1*(4 \ 3)
\\\\
&=(-0.25 \ -0.25)+(0.75 \ 0.75)=(0.5 \ 0.5)
\end{aligned}
$$

所以 $\boldsymbol{w}=(0.5 \ 0.5)$，即：$w_1=0.5, w_2=0.5$，$w_1=w_2$ 只是巧合，因为分界线正好是一条 -45 度的斜线。

### 求解 $b$ 值

对于两类样本的支持向量，存在如下关系：

$$
\begin{cases}
\boldsymbol{w}\boldsymbol{x_i}+b = +1, \quad y_i=+1（正类样本）
\\\\
\boldsymbol{w}\boldsymbol{x_i}+b = -1, \quad  y_i=-1（负类样本）
\end{cases}
\tag{5.5.2}
$$

即样本点距离分界线的距离正好是 1 或 -1。把式 5.5.1 中的两种情况，两边都乘以对应的 $y_i$，得到：

$$
\begin{cases}
1 \cdot (\boldsymbol{w}\boldsymbol{x_i}+b) = 1, \quad y_i=+1（正类样本）
\\\\
(-1) \cdot (\boldsymbol{w}\boldsymbol{x_i}+b) = 1, \quad  y_i=-1（负类样本）
\end{cases}
\tag{5.5.3}
$$

合并两种情况，在支持向量上有：$y_i(\boldsymbol{w} \boldsymbol{x}_i+b)=1$，所以：

$$
b = \frac{1}{y_i}-\boldsymbol{w} \boldsymbol{x}_i \tag{5.5.4}
$$

我们分别用 $p_1$ 和 $p_2$ 两个支持向量验证。

- 负类样本 $p_1$
  
$$
b=\frac{1}{-1} - (0.5 \ 0.5)(1 \ 1)^T=(-1)-1=-2
$$

- 正类样本 $p_2$

$$
b=\frac{1}{1} - (0.5 \ 0.5)(3 \ 3)^T=1-3=-2
$$

所以可以确定最终的 $b=-2$。

### 分界线方程

有了 $\boldsymbol{w}$ 和 $b$ 后，可以得到分界线方程为：

$$
f(\boldsymbol{x}) = \boldsymbol{w} \boldsymbol{x} + b = 0.5x_1+0.5x_2-2=0 \tag{5.5.5}
$$

从式 5.1.4 可知，正类样本位于分界线上方，都应该大于 0；而负类样本位于分界线下方，都应该小于 0。对应到 $f(x)$ 的值为正数或者负数。但是分类结果只有 1 和 -1，不管具体数值是多少，所以有分类决策函数为：

$$
f(\boldsymbol{x})=sign(\boldsymbol{w} \boldsymbol{x}+b) \tag{5.5.6}
$$

其中 $sign()$ 是符号函数，当输入为正数时，返回 1；输入为负数时，返回 -1。

我们用 $p_3(4,3)$ 点验证一下：

$$
f(x)=sign(0.5 \cdot 4 + 0.5 \cdot 3 - 2) = sign(1.5)=1
$$

说明 $p_3$ 点为正类，与实际情况一致。

增加一个 $p_4(2,1)$ 点测试一下：

$$
f(x)= sign(0.5 \cdot 2 + 0.5 \cdot 1 - 2) = sign(-0.5)=-1
$$

说明 $p_4$ 点是负类，与实际情况一致。
