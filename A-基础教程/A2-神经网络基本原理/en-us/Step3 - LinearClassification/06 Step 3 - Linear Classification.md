<!--Copyright © Microsoft Corporation. All rights reserved.
  适用于[License](https://github.com/Microsoft/ai-edu/blob/master/LICENSE.md)版权许可-->

# Step 3  Linear Classification

## Abstract

The classification problem is called logistic regression in many materials because it uses a linear model in linear regression, together with a logistic binary classification function, to construct a classifier. We will call this classification in this book.

One of the essential functions of neural networks is classification. The complexity of classification tasks in the real world is diverse, but everything is the same, and we can use the same pattern of neural networks to handle them.

This section will start with the simplest linear binary classification, including its principle, implementation, training process, inference process, etc., and use visualization to help you better understand these processes.

In this section, we will use our knowledge of binary classification to implement logic AND gates, NAND gates, OR gates, and NOR gates.

When doing binary classification, we generally use the Sigmoid function as the classification function. Can the hyperbolic tangent function, which looks similar to the Sigmoid function, be used as the classification function?We will explore this matter to have a deeper understanding of classification functions, loss functions, and sample labels.

Then we will move to the learning of linear multi-classification. In multi-classification, it can be one-to-one, one-to-many, or many-to-many, so which approach does the neural network use?

The Softmax function is the classification function for the multi-classification problem. By analyzing it, we learn the principles, implementation, and visualization results of multi-classification to understand how the neural network works.
