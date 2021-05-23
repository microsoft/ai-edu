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

To help the reader achieve a level of greatness:

1. Judge which tasks are achievable by machine learning and which are science fiction. Don’t stay an amateur
2. Thoroughly understand the basic theories of neural networks and deep learning
3. Cultivate the ability to solve practical problems by analogies
4. Have the ability to learn more complex models and advanced content alone
5. For the more experienced readers, they can cultivate the ability to develop new models

## Notation and Convention

|Symbol|Meaning|
|---|---|
|$x$|Sample training value|
|$x_1$|The first sample/characteristic value of the sample which will be explained under the context|
|$x_{12},x_{1,2}$|The second eigenvalue of the first sample|
|$X$|Multi-sample training matrix|
|$y$|Sample label value for training|
|$y_1$|Label value of the first sample|
|$Y$|Multi-sample label matrix for training|
|$z$|Linear operation's resulting value|
|$Z$|Linear operation's resulting matrix|
|$Z1$|Linear operation's resulting matrix of the first layer network|
|$\sigma$|Activation function|
|$a$|Activation function's resulting value|
|$A$|Activation function's resulting matrix|
|$A1$|Activation function's resulting matrix of the first layer network|
|$w$|Parameter weight value|
|$w_{12},w_{1,2}$|The weight value at the first row and second column in the parameter weight matrix|
|$w1_{12},w1_{1,2}$|The weight value of the first row and second column in the parameter weight matrix of the first layer network|
|$W$|Parameter weight matrix|
|$W1$|The weight parameter matrix of the first layer network|
|$b$|Offset parameter value|
|$b_1$|The irst offset value in the parameter offset matrix|
|$b2_1$|The first offset value in the parameter offset matrix of the second layer network|
|$B$|Offset parameter matrix (vector)|
|$B1$|Offset parameter matrix of the first layer network (vector)|
|$X^T$|X's transpose matrix|
|$X^{-1}$|X's inverse matrix|
|$loss,loss(w,b)$|One-sample error function|
|$J, J(w,b)$|Multi-sample loss function|
