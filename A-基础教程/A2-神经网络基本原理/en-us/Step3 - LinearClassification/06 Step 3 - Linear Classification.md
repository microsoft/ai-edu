<!--Copyright © Microsoft Corporation. All rights reserved.
  适用于[License](https://github.com/Microsoft/ai-edu/blob/master/LICENSE.md)版权许可-->

# Step 3  Linear Classification

## Abstract

Classification problems are called logistic regressions in many materials because it uses a linear model in linear regression, in addition to a logistic binary classification function, to construct a classifier. We will refer to this as classification in this book.

One of the essential functions of neural networks is classification. Classification tasks in the real world are diverse and complex, but they are all fundamentally the same, and we can use the same pattern of neural networks to handle them.

This section will start with the simplest linear binary classification, including its principle, implementation, training process, inference process, etc., and use visualization to help you better understand these processes.

In this section, we will use our knowledge of binary classification to implement logic AND gates, NAND gates, OR gates, and NOR gates.

When doing binary classification, we generally use the Sigmoid function as the classification function. Can the hyperbolic tangent function, which looks similar to the Sigmoid function, be used as the classification function? We will explore this matter to have a deeper understanding of classification functions, loss functions, and sample labels.

Then we will learn about linear multi-classification. Multi-classification can be one-to-one, one-to-many, or many-to-many, so which approach do neural networks use?

The Softmax function is the classification function for the multi-classification problem. By analyzing it, we learn the principles, implementation, and visualization results of multi-classification to understand how neural networks work.
