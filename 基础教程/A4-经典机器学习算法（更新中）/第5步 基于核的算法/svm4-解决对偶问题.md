https://zhuanlan.zhihu.com/p/154517678

### SVM-4 对偶问题


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

