# NNI 学生项目 2020：Task 3.2 任务指南

## 任务目标

本个项目任务围绕特征工程（Feature Engineering）和自动特征工程（Automated Feature Engineering）开展，包括但不仅限于这些问题：

+ 特征筛选和特征重要性的估计
  + 自动特征工程的核心就是发现不同特征的重要程度。同学们可以使用 NNI 中已有的算法进行特征生成和特征筛选，找出相对重要的特征。当然，我们更希望同学们能沿着前人的脚步，尝试为 NNI 贡献新的 operation ，或特征筛选的算法。
+ 特征搜索空间的设计
  + 表格数据的特征搜索空间的设计：对于表格数据，特征工程起到非常重要的作用。除了搜索空间的设计，处理表格数据的时候还需要特别注意数据预处理、数据编码方式、高阶特征挖掘和基于其他分类的的特征提取等问题。这些问题有些已经有了成熟的解决方案，有些值得我们去挖掘研究。如果能够在处理的时候考虑到它们，可以更加完善和丰富自动特征工程的功能。
  + 典型问题的特征搜索空间的设计：对于不同领域，特征的搜索空间可能有很大的差别。例如，对于 CTR 预估这个问题来说，N阶交叉特征可能起到比较重要作用；对于时间序列预测问题来说，时间相关的特征可能起比较重要的作用。为不同领域设计特征搜索的空间，并提供相关特征抽取的模块，是一件非常有意义的事情。

<br>

## 任务简介

相较于Task 3.1，Task 3.2.1 您需要使用 NNI 或者自己的方法针对表格数据进行更为精细化的特征工程。Task 3.2.2 我们会为您提供推荐系统（Recommender System）、时间序列（Time Series）、点击率预估（CTR  Prediction）三个不同问题的数据，同学们可根据自己的兴趣爱好、学习情况和实际能力选择其中的一个子领域，针对一个特有领域进行特征工程，探索适合自己的进阶之路。

如果您是刚刚入门机器学习的小白，相信经过了 Task 1 与 Task 2 的学习，已经具备了一定的 python 机器学习、深度学习基础，也熟悉了 NNI 的基本用法，本个项目我们将通过 NNI 让您对特征工程有进一步的了解。您可以先简单提取原始数据中的特征来训练模型，然后引入 NNI 工具，尝试特征生成和特征筛选，再次训练模型，并将两次的结果比较分析。

如果您是是身经百战的“调参大师”，并且对于 *[AutoFETuner](https://github.com/SpongebBob/tabular_automl_NNI)* 算法有一定的兴趣，您可以阅读 NNI 特征工程部分的[中文文档](https://nni.readthedocs.io/zh/latest/feature_engineering.html)或[英文文档](https://nni.readthedocs.io/en/latest/feature_engineering.html)，我们十分欢迎您在完成基础任务的前提下成为 NNI 的 contributor ，**您可以尝试在现有的 *[AutoFETuner](https://github.com/SpongebBob/tabular_automl_NNI/blob/master/AutoFEOp.md)* 的基础上增加更多的 operation ，或更多特征筛选的算法**......我们诚挚邀请您反馈，或更多贡献。

只要善于思考，乐于投入，每一位同学都能成为 NNI 的使用者乃至开发者。

准备好了吗？让我们和 NNI 一起开启全新的机器学习之旅吧！

<br>

## Task 3.2.1 表格型数据的进阶任务

### **任务要求**

针对表格类型的数据，在使用 NNI 的时候，您需要思考如何设计特征搜索空间（Search Space），在设计的空间里搜索尝试哪一种组合更好；思考如何设计特征抽取模块，包括但不限于以下几个问题：

1. 数据预处理：数据清洗（包括缺失值的填充、异常值的处理等）和稀疏特征的处理等
2. 数据编码方式：针对类别、数值、多值、时间数据等做不同的处理
3. 高阶特征的挖掘：如何挖掘高阶特征
4. 基于其他分类器的特征提取：如基于KNN的特征、tree的特征

您需要定义自己的模型来训练这些处理后的特征，对于大部分表格数据来说，复杂的深度学习模型都不是最好的选择，所以我们建议同学们进行多种尝试。最后，您需要把处理流程记录在报告中，并比较使用以下两种类型的数据训练模型的效果。

+ 原始数据
+ 使用 NNI 和自己的方法进行特征工程后的数据

我们非常欢迎您成为 NNI 的 contributor ，把处理这些问题过程中用到的全新的 operation 和算法等集成在 NNI 中。

#### **数据下载**

为了验证 Auto Feature Engineering 的方法能在很多不同的数据集上做到较好的泛化性，我们提供了多个数据集，其中覆盖了分类问题、回归问题、multi-label 的问题等。  

+ [电影票房预测：TMDB Box Office Prediction](https://www.kaggle.com/c/tmdb-box-office-prediction/data)
+ [旧金山犯罪分类：San Francisco Crime Classification](https://www.kaggle.com/c/sf-crime/data)
+ [土壤属性预测：Africa Soil Property Prediction Challenge](https://www.kaggle.com/c/afsis-soil-properties/data)

<br>

## Task 3.2.2 复杂型数据的探究任务

在 Task 3.1 和 Task 3.2.1 中，我们尝试了对于表格类型的数据进行特征工程，但是对于特定领域，特征搜索空间设计的侧重点有所不同。下面列举了三个领域中的数据，同学们可以根据自己的兴趣爱好、学习情况和实际能力**选择其中的一个**来完成本小节的任务。

### 任务要求

在完成本小节任务的时候，您可以参考以下的步骤：

1. 针对原始数据做简单的特征提取。
2. 定义机器学习 / 深度学习模型，输入提取后的特征进行模型训练，记录实验结果。
3. 引入 NNI 工具，尝试定义搜索空间，进行特征生成和特征筛选（需要特别关注**特征搜索空间的设计**）。
4. 再次利用这些特征再次训练模型，分析比较两次实验的结果，并将整个过程写入报告中。

### 数据下载

每个领域的第一个数据集为必做题目，后面为选做题目，完成选做题目会有额外加分。

1. 推荐系统（Recommender System）

	+ [产品推荐：Santander Product Recommendation](https://www.kaggle.com/c/santander-product-recommendation/data)
	+ [酒店推荐：Expedia Hotel Recommendations](https://www.kaggle.com/c/expedia-hotel-recommendations/data)
	+ [商品购买预测：Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis/data)

2. 时间序列（Time Series）

	+ [商品销售预测：Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data)
	+ [降雨预测：How Much Did It Rain? II](https://www.kaggle.com/c/how-much-did-it-rain-ii/overview)
	+ [网络流量预测：Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting/data)

3. 点击率预估（CTR  Prediction）

	+ [点击率预测1：Click-Through Rate Prediction](https://www.kaggle.com/c/avazu-ctr-prediction/data)
	+ [点击率预测2：Outbrain Click Prediction](https://www.kaggle.com/c/outbrain-click-prediction/data)
	+ [点击率预测3：Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/data)

<br>

## 说明

关于使用 NNI 的详细教程，请访问 Github 社区：https://github.com/microsoft/nni

我们希望各个队伍独立完成任务，同时，也鼓励各位学员与导师积极沟通，及时反馈，这将有效推动您的项目。

期待大家的好消息！

