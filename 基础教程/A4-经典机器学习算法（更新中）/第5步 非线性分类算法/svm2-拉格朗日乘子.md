https://zhuanlan.zhihu.com/p/154517678


## 拉格朗日乘子法（Lagrange Multiplier）

### 无约束优化问题

对于函数 $f(x,y)=x^2+y^2$，如果求其最小值，我们知道分别对 $x、y$ 求偏导，并令结果为 0，即可以得到极值点结果为 $(0,0)$ 点。

### 等式约束优化问题

如果加一个限制条件，求函数 $f(x,y)$ 在约束条件 $x+y+2=0$ 时的最小值，记作：

$$
\begin{aligned}
\underset{x,y}{\min} & \ f(x,y)=x^2+y^2
\\\\
s.t. & \ x+y+2=0 \qquad \rightarrow h(x,y)
\end{aligned}
\tag{5.2.1}
$$

- 用解方程组的方法

$$
\begin{cases}
    z = x^2+y^2
    \\\\
    x+y+2=0
\end{cases}
\tag{5.2.2}
$$
下面的公式变形为 $y=-x-2$ 带入上面的式子，得到：
$$
z=2x^2+4x+4 \tag{5.2.3}
$$

对$z$求$x$的导数，并令结果为 0：

$$
\nabla_x z = 4x+4 = 0 \tag{5.2.4}
$$

得到：当 $x=-1，y=-1$时，$z=2$ 为最小值。

<img src="./images/5.2.1.png" />

<center>图 5.2.1 相交线及其极值点</center>

结果如图 5.2.1，红色曲线为函数 $f(x,y)$ 与约束平面 $h(x,y)=x+y+2=0$ 的相交线，红色圆点为极值点。此处可以运行 5-2-1.py 来观察实际效果。

注意，$h(x,y)$ 约束条件其实只是 $x/y$ 平面上的一条直线，我们把它“提升”成为一个立平面，这样就和 $f(x,y)$ 形成相交，便于读者理解。

有一种错误的理解是把 $h(x,y)=x+y+2=0$ 看成 $z=x+y+2$，这就变成了一个三维空间中的斜面，与 $f(x,y)$ 的底部相交形成一个倾斜的椭圆，也可以求极值点。但是和原来的约束条件完全是两个不同的问题。

- 拉格朗日乘子法

当约束条件比较复杂时，不能直接解出方程组来，就可以用拉格朗日乘子法。

一个长方体的长宽高分别是 $x,y,z$，表面积是 $s^2$。问当长宽高具体为多少时，体积最大？

如果用解方程组的方法，可以得到：

$$
V = \frac{xy(s^2-2xy)}{2(x+y)}
$$

下一步求极值时，就非常麻烦了。如果参数更多的话，会更麻烦。

下面尝试用拉格朗日乘子法解决，先转化为约束条件：

$$
\begin{aligned}
    \underset{x,y,z}{\max} & \ f(x,y,z) = xyz
    \\\\
    s.t. & \ 2xy+2xz+2yz-s^2=0
\end{aligned}
\tag{5.2.5}
$$

构造拉格朗日函数：

$$
L(x,y,z,\alpha)=f(x,y,z)+\alpha h(x,y,z)=xyz+\alpha(2xy+2xz+2yz-s^2) \tag{5.2.6}
$$

$$
\begin{cases}
    \nabla_x L(x,y,z,\alpha)=yz+2\alpha(y+z)=0
    \\\\
    \nabla_y L(x,y,z,\alpha)=xz+2\alpha(x+z)=0
    \\\\
    \nabla_z L(x,y,z,\alpha)=xy+2\alpha(x+y)=0
    \\\\
    \nabla_{\alpha} L(x,y,z,\alpha)=2xy+2xz+2yz-s^2=0
\end{cases}
\tag{5.2.7}
$$

解方程组 5.2.7 得：

$$x=y=z=\frac{6}{\sqrt{z}}，V_{max}=\frac{s^3}{6\sqrt{6}}$$


### 不等式约束优化问题

问题一：

$$
\begin{aligned}
\underset{x,y}{\min} & \ f(x,y)=x^2+y^2
\\\\
s.t. & \ x+y-1 \le 0 \quad \rightarrow g_1(x,y)
\end{aligned}
\tag{5.2.8}
$$

问题二：

$$
\begin{aligned}
\underset{x,y}{\min} & \ f(x,y)=x^2+y^2
\\\\
s.t. & \ x+y+2 \le 0 \quad \rightarrow g_2(x,y)
\end{aligned}
\tag{5.2.9}
$$


<img src="./images/5.2.2.png" />

<center>图 5.2.2 不等式约束优化</center>

图 5.2.2 展示了两种情况的不等式，左子图是三维图形，包含原函数 $f(x,y)$ 的曲面图和两个约束不等式的立面图，右子图是左图在 $x/y$ 平面上的投影。

两个约束不等式形成了两种情况：

- 问题一：原函数的最优解在不等式的约束允许的区域内，所以没有受到约束的影响

约束不等式 1：$g_1(x,y)=x+y-1 \le 0$，从右子图看，既要求右上方的虚线的左下方的区域为约束允许的区域。

可以使用**广义拉格朗日函数**求解不等式约束问题：

$$
L(x,y,\alpha)=f(x,y)+\alpha g_1(x,y) \tag{5.2.10}
$$

求 $x,y,\alpha$ 的偏导并令其为 0，解出：$\alpha=-1,x=0.5,y=0.5,f(0.5,0.5)=0.5$。

由于原函数 $f(x,y)$ 的最优解 $p_0(x=0,y=0,z=0)$ 在不等式的约束区域允许范围内，所以极值点还是 $p_0$ 点，相当于没有约束。约束边界上的 $p_1(x=0.5,y=0.5,z=0.5)$ 点的 $z$ 值大于 $p_0$ 点的 $z$ 值，所以不是最优解。

- 问题二：最优点在不等式的边界上，改变了原函数的最优解

约束不等式 2：$g_2(x,y)=x+y+2 \le 0$，从右子图看，既要求点划线的左下方的区域为约束允许的区域。

这种情况下，由于原函数是个凸函数，越靠近原点越优，所以最优解应该在约束的边界上，而不是远离约束边界的允许区域内。这就相当于等式约束，那么就依然可以用上面的拉格朗日乘子法来求解：

$$
L(x,y,\alpha)=f(x,y)+\alpha g_2(x,y) \tag{5.2.11}
$$

求 $x,y,\alpha$ 的偏导并令其为 0，解出：$\alpha=2,x=-1,y=-1,f(-1,-1)=2$。

### 同时含有等式和不等式的优化问题

$$
\begin{aligned}
\underset{x,y}{\min}  & \ f(x,y)=x^2+y^2
\\\\
s.t. & \ x-y-2 \le 0 \quad \rightarrow g(x,y)
\\\\
& \ x^2y-3 = 0 \quad \rightarrow h(x,y)
\end{aligned}
\tag{5.2.12}
$$

此时构造拉格朗日函数如：

$$
L(x,y,\alpha,\beta)=f(x,y)+\alpha g(x,y)+\beta h(x,y) \tag{5.2.13}
$$

然后分别求 $x、y、\alpha、\beta$ 的偏导数并令其为 0，联立 4 项等式方程组即可。

$$
\begin{cases}
    \nabla_x L=2x+\alpha+2\beta xy=0
    \\
    \nabla_y L=2y-\alpha+\beta x^2
    \\
    \nabla_{\alpha} L=x-y-2=0
    \\
    \nabla_{\beta} L=x^2y-3=0
\end{cases}
\tag{5.2.14}
$$

### KKT（Karush-Kuhn-Tucker）条件

综合不等式优化中的两种情况考虑：
- 第一种情况相当于在公式 5.2.10 中的 $\alpha=0$，即没有约束，此时 $\alpha g(x,y)=0$ 项没有影响，直接求原函数的最优解即可。

- 第二种情况虽然是不等式，但是最优解在边界上，所以相当于等式约束，即 $g(x,y)=0$，而非 $g(x,y) \le 0$。此时无论 $\alpha$ 为何值，都有 $\alpha g(x,y)=0$。

所以两种情况都满足：
$$
\alpha g(x,y)=0 \tag{5.2.15}
$$

由于 $g(x,y) \le 0$，如果 $\alpha<0$ 的话，则相当于 $\alpha g(x,y) \ge 0$，改变了不等号的方向，不满足拉格朗日乘子法的规则。所以要求：
$$
\alpha \ge 0 \tag{5.2.16}
$$

所以，所谓的 KTT 条件就是可以求解出最优解的必要条件是：

$$
\begin{cases}
    \nabla_{x,y} L = \nabla_{x,y} f(x,y)+\nabla_{x,y}  \alpha g(x,y)=0
    \\\\
    \alpha \ge 0
    \\\\
    \alpha g(x,y)=0
\end{cases}
\tag{5.2.15}
$$


### 思考与练习

1. 用拉格朗日乘子法解决公式 1 提出的问题。
2. 自己动手得出式 12 的解。
3. 运行代码 5-2-2.py 来得到图 5.2.2。
