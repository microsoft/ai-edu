<!--Copyright © Microsoft Corporation. All rights reserved.
  适用于[License](https://github.com/Microsoft/ai-edu/blob/master/LICENSE.md)版权许可-->

# The Second Step - Linear Regression

## Abstract

Using linear regression as the starting point for learning neural networks is a very good choice, because the linear regression problem itself is easier to understand. On the basis of it, gradually adding extra knowledge will form a relatively gentle learning curve, which is the first small step towards a neural network.

A single-layer neural network is actually a neuron, which can perform some linear tasks, such as fitting a straight line. This can be achieved with a single neuron. When this neuron only receives one input, it is a univariate linear regression, which can be visualized on a two-dimensional plane. When receiving multiple variable inputs, it is called multivariate linear regression, which is difficult to visualize, and we usually use pairs of variables to represent it.

When there is more than one variable, the dimensions and values of the two variables may be very different. In this case, we usually need to normalize the sample characteristic data, and then feed the data to the neural network for training, otherwise "indigestion" will occur.
