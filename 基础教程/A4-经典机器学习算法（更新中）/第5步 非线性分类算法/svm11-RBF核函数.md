
## RBF核函数

常用核函数

https://blog.csdn.net/mengjizhiyou/article/details/103437423


|名称|表达式|说明|
|--|--|--|
|线性核|$x_i \cdot x_j  + c$|不能把非线性问题转换成线性问题|
|多项式核|$[\gamma (x_i \cdot x_j) + c]^d$|$\gamma$ 一般取值为 1/特征数，d一般取值为 2，常用形式为 $[\gamma(x_i \cdot x_j)+1]^2$|
|高斯核|$\exp(-\frac{\parallel x-y \parallel^2}{2 \sigma^2})$ |径向基（Radial Basis Function）核，对于样本噪音有较好的抗干扰能力|
|高斯核变种|$\exp (-\gamma \parallel x-y \parallel^2 )$ | $\gamma$ 一般取值为 1/特征数|
|拉普拉斯核|$\exp (- \frac{\parallel x-y \parallel)}{\sigma})$ ||
|指数核|$\exp(-\frac{\parallel x-y \parallel}{2 \sigma^2})$||
|逻辑核|$\tanh(a x\cdot y +c)$|主要用于神经网络;正常 alpha=1/m, m是数据特征维度alpha>0,c<0;非正定核|

- 如果特征的数量大到和样本数量差不多，则选用LR或者线性核的SVM；
- 如果特征的数量小，样本的数量正常，则选用SVM+高斯核函数；
- 如果特征的数量小，而样本的数量很大，则需要手工添加一些特征从而变成第一种情况。
- 实际上使用可以对多个核函数进行测试，选择表现效果最好的核函数。

说明：n为特征数，m为样本个数
（1）若n相对m比较大，如n=10000, m=10~1000, 使用logistic/ SVM（线性核）均可；
（2）若n较小， m中等大小，如n=1~1000, m=10~10000, 使用SVM高斯核；
（3）若n较小，m较大，如n=1~1000, m=50000+, 那么需增加特征，此时用多项式核或高斯核;
(4)更一般：m小，用简单模型；m大用复杂模型。


（1）linear核：无需设置参数；
（2）poly核 ：degree, gamma, coef0；
（3）brf核：gamma；
（4）sigmoid核：gamma, coef0；

具体对于C（惩罚系数）：C越大，越容易过拟合。
gamma：隐含地决定了数据映射到特征空间后的分布，gamma越大，支持向量越少，导致过拟合。


高斯核的映射函数
$$
f(z)=\exp(-\frac{z^2}{2\sigma^2}) \left [1, \sqrt{\frac{2}{1!}}\frac{z}{\sigma}, \sqrt{\frac{2^2}{2!}}(\frac{z}{\sigma})^2,\sqrt{\frac{2^3}{3!}}(\frac{z}{\sigma})^3, \dotsb \right ]
$$


Mercer 定理：任何半正定的函数都可以作为核函数。所谓半正定的函数 $f(x_i,x_j)$，是指拥有训练数据集合 $x_1,x_2,...x_n)$，我们定义一个矩阵的元素 $a_{ij} = f(x_i,x_j)$，这个矩阵式 n*n 的，如果这个矩阵是半正定的，那么 $f(x_i,x_j)$ 就称为半正定的函数。

 

请注意，这个mercer定理不是核函数必要条件，只是一个充分条件，即还有不满足mercer定理的函数也可以是核函数。所谓半正定指的就是核矩阵K的特征值均为非负。




多项式核 κ(x1,x2)=(〈x1,x2+R)d ，显然刚才我们举的例子是这里多项式核的一个特例（R=1,d=2）。虽然比较麻烦，而且没有必要，不过这个核所对应的映射实际上是可以写出来的，该空间的维度是 (m+dd) ，其中 m 是原始空间的维度。
 

高斯核 κ(x1,x2)=exp(−∥x1−x2∥22σ2) ，这个核就是最开始提到过的会将原始空间映射为无穷维空间的那个家伙。不过，如果 σ 选得很大的话，高次特征上的权重实际上衰减得非常快，所以实际上（数值上近似一下）相当于一个低维的子空间；反过来，如果 σ 选得很小，则可以将任意的数据映射为线性可分——当然，这并不一定是好事，因为随之而来的可能是非常严重的过拟合问题。不过，总的来说，通过调控参数σ ，高斯核实际上具有相当高的灵活性，也是使用最广泛的核函数之一。
线性核 κ(x1,x2)=〈x1,x2〉 ，这实际上就是原始空间中的内积。这个核存在的主要目的是使得“映射后空间中的问题”和“映射前空间中的问题”两者在形式上统一起来了。


### 对于高斯核函数的理解

高斯函数

$$
f(x)=\exp \left ( -\frac{(x-\mu)^2}{2 \sigma^2} \right )
$$

二维高斯函数

$$
f(x,y)=\exp \left ( -\frac{(x-\mu_1)^2}{2 \sigma_1^2} - \frac{(y-\mu_2)^2}{2 \sigma_2^2} \right )
$$



高斯核函数的标准定义如式 8 所示。

$$K(x_i,x_j)=e^{-\frac{\parallel x_i-x_j \parallel^2}{2 \sigma^2}} \tag{8}$$




但是由于 $\sigma$ 在分母上，理解起来要绕一下，所以一般写成式 9 的形式，即令：$\gamma = \frac{1}{2\sigma^2}$

$$K(x_i,x_j)=\exp (-\gamma \parallel x_i-x_j \parallel^2 ) \tag{9}$$


一共10个样本，5 正 5 负

### 特征映射

$$
\begin{aligned}
K(x_i,x_j)&=e^{-||x_i-x_j||^2}
\\\\
&=e^{-(x_i \cdot x_i+x_j \cdot x_j -2 x_i \cdot x_j)}
\\\\
&=e^{-x_i^2}e^{-x_j^2}e^{2x_i \cdot x_j} \quad(接下来利用泰勒公式展开第三项)
\\\\
&=e^{-x_i^2}e^{-x_j^2} \left [ \sum_{n=0}^\infty \frac{(2 x_i \cdot x_j)^n}{n!} \right] \quad (接下来展开求和项)
\\\\
&=e^{-x_i^2}e^{-x_j^2} \left [ 1 + \frac{2x_i \cdot x_j}{1!} + \frac{(2x_i \cdot x_j)^2}{2!} + \frac{(2x_i \cdot x_j)^3}{3!} + \cdots \right ] \quad (接下来变成内积形式)
\\\\
&=\left [ e^{-x_i^2}(1 \quad \sqrt{\frac{2}{1!}}x_i \quad \sqrt{\frac{2^2}{2!}}x_i^2 \quad \sqrt{\frac{2^3}{3!}}x_i^3 \cdots) \right ] \cdot \left [ e^{-x_j^2}(1 \quad \sqrt{\frac{2}{1!}}x_j \quad \sqrt{\frac{2^2}{2!}}x_j^2 \quad \sqrt{\frac{2^3}{3!}}x_j^3 \cdots) \right ]
\\\\
&=\phi(x_i) \cdot \phi(x_j)
\end{aligned}
$$

$$
\phi(x)=e^{-x^2}
\begin{pmatrix}
1 \quad \sqrt{\frac{2}{1!}}x \quad \sqrt{\frac{2^2}{2!}}x^2 \quad \sqrt{\frac{2^3}{3!}}x^3 \cdots   
\end{pmatrix}
$$




把所有的样本都看作是地标（landmark），构造10x10的矩阵，


$$
FeatureMap=
\begin{pmatrix}
e^{-\gamma\parallel x_1 - x_1 \parallel^2} & e^{-\gamma\parallel x_1 - x_2 \parallel^2} & \cdots & e^{-\gamma\parallel x_1 - x_{10} \parallel^2}
\\\\
e^{-\gamma\parallel x_2 - x_1 \parallel^2} & e^{-\gamma\parallel x_2 - x_2 \parallel^2} & \cdots & e^{-\gamma\parallel x_2 - x_{10} \parallel^2}
\\\\
\vdots & \vdots &  \ddots & \vdots
\\\\
e^{-\gamma\parallel x_{10} - x_1 \parallel^2} & e^{-\gamma\parallel x_{10} - x_2 \parallel^2} & \cdots & e^{-\gamma\parallel x_{10} - x_{10} \parallel^2}
\end{pmatrix}
=
\begin{pmatrix}
1 & e^{-\gamma\parallel x_1 - x_2 \parallel^2} & \cdots & e^{-\gamma\parallel x_1 - x_{10} \parallel^2}
\\\\
e^{-\gamma\parallel x_2 - x_1 \parallel^2} & 1 & \cdots & e^{-\gamma\parallel x_2 - x_{10} \parallel^2}
\\\\
\vdots & \vdots &  \ddots & \vdots
\\\\
e^{-\gamma\parallel x_{10} - x_1 \parallel^2} & e^{-\gamma\parallel x_{10} - x_2 \parallel^2} & \cdots & 1
\end{pmatrix}
$$


### SVC 线性分类器

验证FeatureMap的线性可分性
   
svc(kernal='linear', C=2)

得到score=1.0，表明所有样本点都分类正确

