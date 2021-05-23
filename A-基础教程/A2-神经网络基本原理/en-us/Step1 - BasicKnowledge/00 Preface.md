<!--Copyright © Microsoft Corporation. All rights reserved.
  适用于[License](https://github.com/Microsoft/ai-edu/blob/master/LICENSE.md)版权许可-->

# Preface

**"What I cannot create, I do not understand." (American physicist Richard Feynman)**

In today's technologically advanced world, a lot of knowledge can be found on the Internet. However, the quality of blogs/articles is questionable, its focus is not clear, or it is directly copied from other people’s blogs. This situation makes it difficult for most beginners to learn. They may even be led astray, increasing the steepness of the learning curve. Of course, there are creators who are very responsible. The quality of their articles may be very high, but its scope is not enough. When you are enjoying yourself, there are no follow-up chapters and so it cannot be a good learning system.

Beginners can choose to read textbooks or books on theory, but the chicken and the egg problem arises: If you don’t understand, then after reading you still won’t understand the theory; if you understand, then you don’t need to read the theory. This is a shortcoming of some textbooks and theoretical books.

The author has seen professor Andrew Ng's (Wu Enda) classes. The theoretical knowledge is simple yet deep, and is very clear. Although there are very few code examples, I still strongly recommend looking into Andrew Ng's classes. The author's experience is that Andrew Ng's videos can be saved on a mobile device, and you can watch and learn on your own time.

There are external reasources that you can use to study deep learning online. The author took part in several group purchases. The teachers and teaching assistants are generally very responsible. Lastly, you can rewatch the videos and download the powerpoint course materials. These courses mainly focus on engineering projects, and explain the use of deep learning frameworks and tools. That is, teaching you how to use helpful tools for modeling, training, etc. But for beginners, understanding a new concept may require a lot of previously attained knowledge. An extremely steep learning curve can be frustrating. Maybe someone knows X but doesn't know why. They could end up as an assistant engineer, and their career development ends up being restricted.

There is a proverb: Teach him how to fish and you feed him for his life time. After experiencing the mentioned learning experience, the author, who is a programmer, eagerly feels that a learning experience should be done in "learning by doing". By reproducing some basic theories in writing code, you can deeply understand its meaning. Then you can expand and extend so readers have the ability to draw inferences.

The author, who has summarized their learning experience has also summarized the introductory knowledge of deep learning into 9 steps. These steps are referred to as the 9-step learning method:

1. Basic concepts
2. Linear regression
3. Linear classification
4. Nonlinear regression
5. Non-linear classification
6. Model inference and deployment
7. Deep neural networks
8. Convolutional neural networks
9. Recurrent neural networks

Many books I have seen start directly from Step 7. Their assumption is that the reader has mastered previous knowledge. However, for beginners starting from scratch, this assumption is not correct.

In the following explanations, we will generally use the following methods:

1. Ask the question: First ask a hypothetical question related to reality. In order to move from simple stuff to deeper stuff, these questions should not complicated. They are simplified versions of actual engineering problems.
2. Solution: Use the knowledge of neural networks to solve these problems. Start from simple and basic models and build step by step to achieve a complex model.
3. Principle analysis: Use basic physics concepts or mathematical tools to understand how neural networks work.
4. Visual understanding: Visualization is an important aspect of learning new knowledge. As we use simple examples, they can easily be visualized.

Principle analysis and visual understanding are also features of this book - trying to make neural networks interpretable instead of blindly using them.

This is also a very important point. We have supporting Python code. In addition to some necessary scientific computing libraries and plotting libraries, such as NumPy and Matplotlib, we did not use any existing deep learning frameworks, but led everyone from scratch. Start to build your own simple knowledge system to understand the many knowledge points in deep learning, and slowly scale to more complex ones.

For our friends who have no experience in Python, reading the sample code can also help you learn Python, which can kill two birds with one stone. As the difficulty of a problem deepens, the code will grow as well. The before and after have a connection much like inheritance. The final code will form a small framework, which I call a Mini-Framework. They can be created by calling building block functions to build deep learning components.

The code has been written and debugged by the author himself. Each chapter can run independently, and the results described in the relevant chapters can be obtained, including printouts and graphical output.

In addition, for ease of understanding, the author draws a large number of schematic diagrams. The number totals to more than 10 times the amount of similar books. A picture is worth a thousand words. I believe that everyone will quickly and completely understand the knowledge points I want to share through these diagrams. This way, everyone can start from the real "zero", have a basic understanding of neural networks and deep learning, and be able to practice.

Requirements for readers:

1. Have learned linear algebra and differentiation in advanced mathematics
2. Have programming fundamentals. You don't need to know Python because you can learn it from the sample code
3. Thinking + hands-on learning mentality

可以帮助读者达到的水平：

1. 可以判断哪些任务是机器学习可以实现的，哪些是科学幻想，不说外行话
2. 深刻了解神经网络和深度学习的基本理论
3. 培养举一反三的解决实际问题的能力
4. 得到自学更复杂模型和更高级内容的能力
5. 对于天资好的读者，可以培养研发新模型的能力

## 符号约定

|符号|含义|
|---|---|
|$x$|训练用样本值|
|$x_1$|第一个样本或样本的第一个特征值，在上下文中会有说明|
|$x_{12},x_{1,2}$|第1个样本的第2个特征值|
|$X$|训练用多样本矩阵|
|$y$|训练用样本标签值|
|$y_1$|第一个样本的标签值|
|$Y$|训练用多样本标签矩阵|
|$z$|线性运算的结果值|
|$Z$|线性运算的结果矩阵|
|$Z1$|第一层网络的线性运算结果矩阵|
|$\sigma$|激活函数|
|$a$|激活函数结果值|
|$A$|激活函数结果矩阵|
|$A1$|第一层网络的激活函数结果矩阵|
|$w$|权重参数值|
|$w_{12},w_{1,2}$|权重参数矩阵中的第1行第2列的权重值|
|$w1_{12},w1_{1,2}$|第一层网络的权重参数矩阵中的第1行第2列的权重值|
|$W$|权重参数矩阵|
|$W1$|第一层网络的权重参数矩阵|
|$b$|偏移参数值|
|$b_1$|偏移参数矩阵中的第1个偏移值|
|$b2_1$|第二层网络的偏移参数矩阵中的第1个偏移值|
|$B$|偏移参数矩阵（向量）|
|$B1$|第一层网络的偏移参数矩阵（向量）|
|$X^T$|X的转置矩阵|
|$X^{-1}$|X的逆矩阵|
|$loss,loss(w,b)$|单样本误差函数|
|$J, J(w,b)$|多样本损失函数|
