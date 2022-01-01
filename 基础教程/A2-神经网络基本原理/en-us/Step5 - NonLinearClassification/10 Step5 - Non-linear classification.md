<!--Copyright © Microsoft Corporation. All rights reserved.
  适用于[License](https://github.com/Microsoft/ai-edu/blob/master/LICENSE.md)版权许可-->

# Step5  Non-linear classification

## Abstract

We learned linear classification in step 3, and in this section, we will study more complex classification problems. For example, many years ago, two famous scholars proved that perceptrons cannot solve the XOR problem in logic, thus bringing the research field of perceptrons to long-term stagnation. We will be using two-layer networks to solve the XOR problem.

The XOR problem is a simple binary classification problem because there are only 4 sample data. We will use more complex data samples to learn the non-linear multi-classification problem and understand how its works.

We will then use a slightly more complex binary classification example to illustrate how a neural network can transform a linearly non-separable problem into a linearly separable one in the two-dimensional plane by a magical linear transformation plus activation function compression.

After solving the binary classification problem, we will learn how to solve the more complex triple classification problem, where multiple neurons must be used in the hidden layer to complete the classification task due to the complexity of the sample.

Finally, we will build a three-layer neural network to solve the MNIST handwritten digit recognition problem and learn to use gradient checking to help us test the correctness of the backpropagation code.

The use of datasets is a fundamental skill in deep learning. The development set, validation set, and test set are used wisely to get the ideal model with high generalization ability.
