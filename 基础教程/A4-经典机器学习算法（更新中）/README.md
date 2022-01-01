经典的机器学习算法仍然在工程实践中发挥着重大作用。对此有兴趣的读者可以为此贡献自己的教案，比如 SVM, BoostTree, KNN, 等等等等。

需要有算法分析、算法实现、实际案例，用python完成。

您可以任选一个算法，提交您的PR。

使用sckiti learn or Azure ML

https://docs.microsoft.com/zh-cn/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-15963-cxa


- Regression
- KNN
- SVM
  - linear SVM
  - RBF SVM
- Decision Tree
- Navie Bayes
- Boosting
- Clustering
- PCA
- Gaussian Process
- Random Forest
- AdaBoost
- QDA (Quadratic Discrination Analysis)

常见的机器学习算法：

1）回归算法：回归算法是试图采用对误差的衡量来探索变量之间的关系的一类算法。回归算法是统计机器学习的利器。 常见的回归算法包括：最小二乘法（Ordinary Least Square），逐步式回归（Stepwise Regression），多元自适应回归样条（Multivariate Adaptive Regression Splines）以及本地散点平滑估计（Locally Estimated Scatterplot Smoothing）。

2）基于实例的算法：基于实例的算法常常用来对决策问题建立模型，这样的模型常常先选取一批样本数据，然后根据某些近似性把新数据与样本数据进行比较。通过这种方式来寻找最佳的匹配。因此，基于实例的算法常常也被称为“赢家通吃”学习或者“基于记忆的学习”。常见的算法包括 k-Nearest Neighbor(KNN), 学习矢量量化（Learning Vector Quantization， LVQ），以及自组织映射算法（Self-Organizing Map，SOM）。

3）决策树学习：决策树算法根据数据的属性采用树状结构建立决策模型， 决策树模型常常用来解决分类和回归问题。常见的算法包括：分类及回归树（Classification And Regression Tree，CART），ID3 (Iterative Dichotomiser 3)，C4.5，Chi-squared Automatic Interaction Detection(CHAID), Decision Stump, 随机森林（Random Forest），多元自适应回归样条（MARS）以及梯度推进机（Gradient Boosting Machine，GBM）。

4）贝叶斯方法：贝叶斯方法算法是基于贝叶斯定理的一类算法，主要用来解决分类和回归问题。常见算法包括：朴素贝叶斯算法，平均单依赖估计（Averaged One-Dependence Estimators，AODE），以及Bayesian Belief Network（BBN）。

5）基于核的算法：基于核的算法中最著名的莫过于支持向量机（SVM）了。基于核的算法把输入数据映射到一个高阶的向量空间，在这些高阶向量空间里，有些分类或者回归问题能够更容易的解决。常见的基于核的算法包括：支持向量机（Support Vector Machine，SVM）， 径向基函数（Radial Basis Function，RBF)，以及线性判别分析（Linear Discriminate Analysis，LDA)等。

6）聚类算法：聚类，就像回归一样，有时候人们描述的是一类问题，有时候描述的是一类算法。聚类算法通常按照中心点或者分层的方式对输入数据进行归并。所以的聚类算法都试图找到数据的内在结构，以便按照最大的共同点将数据进行归类。常见的聚类算法包括 k-Means算法以及期望最大化算法（Expectation Maximization，EM）。

7）降低维度算法：像聚类算法一样，降低维度算法试图分析数据的内在结构，不过降低维度算法是以非监督学习的方式试图利用较少的信息来归纳或者解释数据。这类算法可以用于高维数据的可视化或者用来简化数据以便监督式学习使用。常见的算法包括：主成份分析（Principle Component Analysis，PCA），偏最小二乘回归（Partial Least Square Regression，PLS），Sammon映射，多维尺度（Multi-Dimensional Scaling, MDS）, 投影追踪（Projection Pursuit）等。

8）关联规则学习：关联规则学习通过寻找最能够解释数据变量之间关系的规则，来找出大量多元数据集中有用的关联规则。常见算法包括 Apriori算法和Eclat算法等。

9）集成算法：集成算法用一些相对较弱的学习模型独立地就同样的样本进行训练，然后把结果整合起来进行整体预测。集成算法的主要难点在于究竟集成哪些独立的较弱的学习模型以及如何把学习结果整合起来。这是一类非常强大的算法，同时也非常流行。常见的算法包括：Boosting，Bootstrapped Aggregation（Bagging），AdaBoost，堆叠泛化（Stacked Generalization，Blending），梯度推进机（Gradient Boosting Machine, GBM），随机森林（Random Forest）。





逻辑回归在什么地方写？是单独一章？属于二分类问题

方差偏差单独写？
https://blog.csdn.net/cprimesplus/article/details/97178227

安斯库姆四重奏
https://www.zhihu.com/question/67493742


线性回归是统计学中最基础的数学模型，几乎各个学科的研究中都能看到线性回归的影子，比如量化金融、计量经济学等；当前炙手可热的深度学习也一定程度构建在线性回归基础上。因此，每个人都有必要了解线性回归的原理。

线性回归的一种最直观解法是最小二乘法，其损失函数是误差的平方，具有最小值点，可以通过解矩阵方程求得这个最小值。尽管推导过程有大量数学符号，线性回归从数学上来讲并不复杂，有微积分和线性代数基础的朋友都可以弄清其原理。
