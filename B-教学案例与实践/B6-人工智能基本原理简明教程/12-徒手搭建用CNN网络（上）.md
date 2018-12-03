# 如何建立深度学习模型（上）

学了很多关于深度学习的直观印象和理论知识了，那么，如何去建立一个深度学习模型呢？这要分成两种情况，使用深度学习框架和不使用深度学习框架。

用一个形象的说法来描述一下深度学习框架，深度学习框架就是提供给你使用的积木。比如要搭一座积木小屋，只需要选择合适的积木，把他们按照次序堆叠起来，就可以得到想要的积木小屋。深度学习模型就是这样一个积木小屋，构建这座小屋的积木就是深度学习框架提供的封装好的各种类型的layer的调用函数，比如卷积层，relu层等等。

针对mnist，使用tensorflow构建的代码（来自[samples-for-ai](https://github.com/Microsoft/samples-for-ai)），是这个样子的

```python
def model(data, train=False):

    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].

    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')

    # Bias and rectified linear non-linearity.

    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.

    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')

    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))

    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.

    pool_shape = pool.get_shape().as_list()

    reshape = tf.reshape(
        pool,
        [tf.shape(pool)[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.

    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.

    if train:

      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)

    return tf.matmul(hidden, fc2_weights) + fc2_biases
```

在上面这份代码中，我们的积木玩具是`tf.nn.conv2d`,`tf.nn.relu`,`tf.nn.max_pool`,`tf.nn.dropout`这样一些函数。通过控制每个函数输入和输出的尺寸，就像有接口的积木，相同接口尺寸的积木可以严密的搭到一起。这些函数通过彼此之间输入和输出的尺寸很方便的结合到一起，从而构成一张计算图。之后要做的就是提供有标签的数据进行计算和反向传播的训练了。

但是如果不使用深度学习框架呢？

不使用深度学习框架，我们需要寻找一个可以替代的工具。就像我们现在要找一个合适的工具将手头的原材料打磨成合适搭建的积木。从之间文章的大段公式推导中，我们可以发现，深度学习的很大一部分基础是构建在线性代数的基础上的，准确来说，很大一部分运算的基础是矩阵有关的运算。

python中和矩阵运算有关的函数库是什么呢？numpy啊！所以，我们可不可以使用numpy来搭建一个深度学习的模型呢？

仿照搭积木的过程，现在我们有了目标的房子图纸（模型），原材料（numpy），需要自己去是实现的也就是用来搭建的这些“积木”了。

如何使用numpy来构建这样的积木呢？

我们知道，深度学习有两个关键的方面，一个是前向传播，一个是反向传播，反向传播的关键点就是梯度计算和模型更新了。所以，我们构建的积木的关键点也就是实现每一种layer的前向和反向传播计算方法。这里简单起见，我们不做一个通用的各层结构的实现，而是单纯考虑像上面mnist中那样搭积木。

在激活函数和损失函数一节中，我们已经了解过激活函数和损失函数的前向和反向传播公式和代码实现，剩下需要完成的积木就是`tf.nn.dropout`，`tf.nn.max_pool`，`tf.nn.conv2D`和最后的`tf.matmul(hidden, fc2_weights) + fc2_biases`这样几个模块的内容了。

从简单的入手。

### `tf.nn.dropout`

+ 函数作用：  
  通过随机舍弃一定比例的神经元来防止网络过拟合。也就是说，按照一定的比例，将部分神经元的输出置成零，其余神经元输出则会被乘以一个和比例有关的倍数，以保持输出的总的大小和不使用dropout层时基本一致。  
  举个例子，如果函数的输出结果是这个样子的  
  
  $$
  \begin{matrix}
   2 & 2\\
   2 & 2
  \end{matrix}
  $$  
  保持50%的神经元，也就是说，会有一半的神经元被置成零，保留的一半神经元会被乘以
  
  $$\frac{1}{\frac{1}{2}} = 2$$
  
  以保持整体输出的量级。也就说，函数的输出会变成

  $$
  \begin{matrix}
   4 & 0\\
   0 & 4
  \end{matrix}
  $$  

+ 前向传播：  
  按照输入的期望保持概率保留神经元，将不被保留的神经元的输出暂时置零，被保留的神经元的输出乘以

  $$\frac{1}{probability}$$

+ 反向传播：  
  显然，被舍弃的神经元在本次前向传播过程中并没有发挥作用，也就是说，被舍弃的神经元被暂时性的和整个神经网络隔开了，和他们有关的权重也应该被保持原样而非更新。怎么让被舍弃的神经元暂时不更新呢？考虑到神经网络的参数更新是通过梯度计算实现的，梯度传播是依靠各层之间的链式法则来实现。如果有一层针对某个神经元传递的梯度是零是零会怎么样呢？那么这个神经元感知到的梯度是零，也就是说，按照梯度传播更新的方法，这个神经元将不会在这次反向传播过程中不被更新，其余神经元可以继续更新。

+ 代码实现：

  ```python
  class Cdropout(object):

    def __init__(self, inputSize, prob):

        # 记录输出数据尺寸和保持神经元的概率

        self.outputShape = inputSize

        self.prob = prob

    def forward(self, data, train=False):

        self.data = data

        # dropout层只需要在训练过程中发挥作用

        self.train = train

        # 决定每个神经元是否要保留

        self.mask = np.random.rand(
            self.outputShape[0], self.outputShape[1]) > self.prob

        if train:

            return np.multiply(data, self.mask) / (1 - self.prob)

        else:

            return data

    def gradient(self, preError):

        if self.train:

            # 被保留的神经元的梯度会继续传递，被置零的神经元的梯度则会同样的被置成零

            return np.multiply(preError, self.mask)

        else:

            return preError
  ```

### `tf.nn.max_pool`

+ 函数作用：  
  pooling层的作用是在使用一个小区域在整个输出的矩阵上进行滑动，将这个小区域的最大值输出。这个做法有什么好处呢？用例子说话吧，假设我们有这样两个矩阵
  
  $$
  \begin{pmatrix}
   10 & 8 & 9 & 7\\
   2 & 2 & 12 & 3\\
   20 & 15 & 8 & 9\\
   13 & 21 & 23 & 12
  \end{pmatrix} \tag{1}
  $$  
  $$
  \begin{pmatrix}
   0 & 10 & 8 & 9\\
   0 & 2 & 2 & 12\\
   0 & 20 & 15 & 8\\
   0 & 13 & 21 & 23
  \end{pmatrix} \tag{2}
  $$  
  
  矩阵2是由矩阵1向右平移了一位并且用零去填充的一个结果。如果我们使用一个尺寸大小是2\*2，每次移动距离（步长）是2的max_pool会怎么样呢？

  用图来形象化这个过程

  <img src="./Images/12&13/matrix1.png" width="200">

  在这张图中，颜色相同的块表示属于同一个小区域，在每个小区域进行取区域中最大值的操作，得到结果如下

  $$
  \begin{pmatrix}
   10& 12\\
   21 & 23
  \end{pmatrix} \tag{1}
  $$  
  $$
  \begin{pmatrix}
   10& 12\\
   21 & 23
  \end{pmatrix} \tag{2}
  $$  
  
  矩阵1和矩阵2经过这样一个max_pool操作之后得到的结果是一样的！可以看到，输入的平移并没有给输出带来影响。这也是pooling层的作用之一，可以将忽略部分平移或者由噪声对输入造成的影响。也就是max_pool对平移这样的操作有一定的忍耐性。当然，有利有弊。在增加了一定对噪声和平移的容忍能力的同时，这个函数也会造成结果的分辨率进一步下降，当需要在一张尺度很大的图寻找一个小目标的时候，这一个特性就是缺点了。

+ 前向传播  
  从上面的例子不难看出，在max_pool的操作中，我们是使用一个固定尺寸的滑动窗口在矩阵上进行滑动，将窗口中的最大值作为输出结果输出。直观的理解就是，将输入矩阵进行分块，每一块只保留最大值，其余各项的值置为NULL，将矩阵中所有NULL去掉作为输出结果。所以，这个函数的关键参数就是窗口的尺寸，以及窗口在矩阵上每次滑动的距离，也就是窗口移动的步长。

+ 反向传播  
  和dropout情况下类似，在dropout中，只有保留下来的神经元才可以继续传播梯度，在这里，只有一个区域中的最大值才可以继续去进行梯度传播，其余神经元并没有对下一层有任何贡献，也就是说，他们所接收到的反向传播回来的梯度应当是零。

+ 代码实现：

  ```python
  class CmaxPool(object):

    def __init__(self, inputSize, kernelSize=2, stride=2):

        # 记录输入输出尺寸和步长等数据信息

        self.shape = inputSize
        self.kernelSize = kernelSize
        self.stride = stride
        self.outputChannels = inputSize[-1]
        self.batchSize = inputSize[0]

        # 根据输入尺寸和步长等计算输出尺寸

        len = (self.shape[1] - kernelSize) // stride + 1
        self.output = np.zeros((self.batchSize, len, len, self.outputChannels))
        self.outputShape = self.output.shape

    def forward(self, image):

        # 用于记录输入中每一块最大值的位置

        self.memory = np.zeros(image.shape)

        for b in range(self.batchSize):
            for c in range(self.outputChannels):
                # 以stride作为步长，kernelSize作为小区域的尺寸进行取值
                for i in range(0, image.shape[1], self.stride):
                    for j in range(0, image.shape[2], self.stride):

                        self.output[b, i // self.stride, j // self.stride, c] = np.max(
                            image[b, i:i + self.kernelSize, j:j + self.kernelSize, c]
                        )

                        # 找出每一块的最大值的位置
                        index = np.argmax(image[b, i:i + self.kernelSize, j:j+ self.kernelSize, c])
                        self.memory[b, i + index % self.stride, j + index % self.stride, c] = 1

        return self.output

    def gradient(self, preError):

        # 将返回的误差信息每个行列沿输入的长和宽方向重复stride次，将误差还原到输入尺寸

        preError = np.repeat(preError, self.stride, axis=1)
        preError = np.repeat(preError, self.stride, axis=2)

        return np.multiply(self.memory, preError)
  ```

### `tf.matmul(hidden, fc2_weights) + fc2_biases`

+ 函数作用：

  这个函数就是代表了一个全连接层的函数，可以在神经网络中起到一个分类器的作用。就是接收之前的各层传递来的信息，将他们用一种加权的方式投影到标签所对应的空间中。函数的形式就是一个简单的矩阵乘法另外加上偏置。

+ 前向传播：

  这样一个全连接层的前向传播其实就是标题的函数形式。通过将输入和本层的权重进行矩阵乘法，之后再添加偏置，即可构成本层的输出。

+ 反向传播：

  这里的反向传播过程就是反向传播中提到的四大公式，
  
  $$\delta^{L} = \nabla_{a}C \odot \sigma_{'}(Z^L) \tag{1}$$
  $$\delta^{l} = ((W^{l + 1})^T\delta^{l+1})\odot\sigma_{'}(Z^l) \tag{2}$$
  $$\frac{\partial{C}}{\partial{b_j^l}} = \delta_j^l \tag{3}$$
  $$\frac{\partial{C}}{\partial{w_{jk}^{l}}} = a_k^{l-1}\delta_j^l \tag{4}$$
  
  我们可以直接通过将权重参数矩阵转置的方式，和接收到的误差矩阵作矩阵乘法，就可以继续进行反向传播过程。简单的来说，就是执行反向传播四大公式的公式二

  $$\delta^{l} = ((W^{l + 1})^T\delta^{l+1})\odot\sigma_{'}(Z^l)$$

+ 示例代码：

  ```python
  class Cfc(object):
    def __init__(self, inputSize, outputNum=2):

        # 初始化全连接层，根据输入输出生成权重矩阵和偏置矩阵
        self.shape = inputSize
        self.batch = inputSize[0]

        self.weights = np.random.standard_normal((reduce(lambda x, y: x * y, self.shape[1:]), outputNum)) / 100

        self.bias = np.random.standard_normal(outputNum) / 100
        self.output = np.zeros((self.batch, outputNum))
        self.weightsGrad = np.zeros(self.weights.shape)
        self.biasGrad = np.zeros(self.bias.shape)
        self.outputShape = self.output.shape

    def forward(self, image):

        # 将输出处理成列向量，预设的全连接层输入的形式

        image = np.reshape(image, [self.batch, -1])

        # 执行矩阵乘法和偏置

        fcout = np.dot(image, self.weights) + self.bias
        self.output = fcout
        self.image = image

        return fcout

    def gradient(self, preError):

        # 根据回传的误差和反向传播的四大公式计算本层梯度和继续传递的误差

        for i in range(self.batch):
            imagei = self.image[i][:, np.newaxis]
            preErrori = preError[i][:, np.newaxis].T
            self.weightsGrad = self.weightsGrad + np.dot(imagei, preErrori)
            self.biasGrad = self.biasGrad + np.reshape(preErrori, self.biasGrad.shape)

        return np.reshape(np.dot(preError, self.weights.T), self.shape)

    def backward(self, learningRate=0.001, weightsDecay=0.004):

        # 根据梯度和学习率等进行权重和偏置的更新，并且将梯度矩阵置零

        weights = (1 - weightsDecay) * self.weights
        bias = (1 - weightsDecay) * self.bias
        self.weights = weights - learningRate * self.weightsGrad
        self.bias = bias - learningRate * self.biasGrad
        self.weightsGrad = np.zeros(self.weights.shape)
        self.biasGrad = np.zeros(self.bias.shape)
  ```

  上篇到此结束~我们将在下篇介绍卷积层和损失函数的有关实现~