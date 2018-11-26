# 徒手搭建CNN网络(下)

在这一部分，您将看到有关卷积层和softmax加crossentropy的原理和实现~

### `tf.nn.conv2D`

+ 函数作用:  
  这个函数就是传说中的卷积层函数了。这个函数的作用呢，就是通过卷积操作对输入的数据进行处理，提取图片中隐藏的特征。卷积操作的过程和max_pool处很类似，也是通过对一个小区域进行操作得到我们所期望的结果。不同于max_pool中简单的选取最大值，卷积操作中对这样一个小区域执行的操作是由卷积核决定的。不同卷积核对应提取不同的特征。
  
  不过为什么不采用全连接层来实现这样一个特征提取操作呢？一个主要的考虑就是参数的数量问题。在卷积层的特征提取操作中，可以参考滑动窗口的想法，一个卷积核是通过在整个输入上滑动来实现让所有输入共享同样的卷积核参数的，而全连接层则需要每个神经元之间互相都要连接，这样的参数数目是非常多的。
  
  做一个简单的算术，一个4\*4大小的输入，如果使用一个3\*3的卷积核，每次移动步长是1，不在输入周围补零的情况下，可以得到一个多大的输出呢？

  我们来模拟一下卷积核移动的情况

  <img src="./Establish/convprocess.png" width="400">
  
  从图上可以看出，一共有4个这样的小区域会和一个3\*3的卷积核进行卷积，对应的也就是2\*2共4个输出。假设卷积核是这个样子的

   $$ kernel = 
  \begin{pmatrix}
   1& 1 & 1\\
   1 & 1 & 1\\
   1 & 1 & 1
  \end{pmatrix}
  $$

  对每个小区域进行卷积的公式是
  
  $$\sum\limits_{i,j = 0}\limits^{3}kernel_{i,j}*data_{i,j}$$
  
  i,j是每个区域内相对于其左上角的偏移，在当前使用的卷积核的情况下，等于每个区域都是该区域内所有元素值的求和，也就是说，输出的形式是:

  $$ kernel = 
  \begin{pmatrix}
   86& 73\\
   116 & 105
  \end{pmatrix}
  $$

  同样的计算过程如果使用全连接层来做需要多少参数呢？输入的神经元个数是$5*5=25$个，输出是$2*2=4$个，所以一共需要的参数是$25*4=100$个，远远多于卷积层的9个参数。

  从上面的过程中，我们看到了卷积层的计算步骤和比起全连接层节省参数的一个优点。共享参数这样的事情，除了节省参数，还会有什么优点呢？
  
  不知道大家有没有思考过这样一个问题：在很多的物体识别应用中，为什么不管物体在哪个位置都能被网络识别出来呢？

  当然，一方面是可能我们用于训练的数据集中包括了物体在各种各样位置的图片，另一方面，请大家想一想这样一个共享参数的卷积层的作用。这样一个卷积核提取出来的特征和物体所在的位置有关系吗？因为参数是整个输入共享的，所以无论物体在图片中的什么位置，只要存在这个物体，都可以让这种卷积核在对应的位置产生对应的输出。这也就是卷积神经网络具有平移不变性的一个解释。

+ 前向传播

  先看tensorflow中对这个函数调用的接口，

  ```python
  tf.nn.conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=True,
    data_format='NHWC',
    dilations=[1, 1, 1, 1],
    name=None
  )
  ```

  看比较常用的几个参数，input代表着输入卷积层的数据，filter代表了卷积核，strides代表的是卷积核移动的步长，padding选项表示是否要对输出进行周围补零的操作从而使输出和输入有同样的长宽尺寸。

  其中，卷积核filter的尺寸是`[filter_height, filter_width, in_channels, out_channels]`，in_channels需要和输入的通道数相对应，因为一个卷积核是同时对所有输入通道进行操作的，out_channels是输出的通道数，也是该层拥有的卷积核的数量，一个卷积核负责一个通道的输出。

  在这里，为了和示例中采用的mnist代码相对应，我们只来看一下通过补零使输入和输出尺寸相同的算法步骤。

  首先，根据输入尺寸和卷积核大小，我们要对输入进行补零的操作，也就是说，扩大输入尺寸，来使经过卷积层之后得到的尺寸恰好和输入的尺寸相同。这样一个操作叫做padding。我们应该怎么处理输入数据才能使输入输出尺寸一致呢？对于上面的例子，很容易发现，对两侧各补一行或者一列零就可以啦！也就是这个样子：

  <img src="./Establish/padding.png" width="300">

  最外面一圈的零是补充的结果，使用这样一个矩阵和卷积核去执行卷积操作，就可以得到和原始输入相同尺寸的输出大小。

  那么，具体这个零应该怎么补充呢？在不添加padding的情况下，输出的尺寸和输入的关系是：

  $$ out_{size} = \frac{input_{size} - kernel_{size}}{strides} - 1$$

  在这里，我们希望$out_{size} = input_{size}$，也就是说，经过padding之后的输入尺寸$input_{size}^{'}$应当满足：

  $$ input_{size} = \frac{input_{size}^{'} - kernel_{size}}{strides} - 1$$

  经过简单的运算之后，可以得到

  $$ input_{size}^{'} = (input_{size} - 1) * strides + kernel_{size}$$

  因为对于每一个维度来说，padding会在输入的两侧同时进行补零操作，所以padding的尺寸是:

  $$ padding = \frac{input_{size}^{'} - input_{size}}{2} = \frac{(input_{size} - 1) * strides + kernel_{size} - input_{size}}{2}$$
  
  在这里，我们使用步长为1的卷积，也就是$strides = 1$，那么，padding的大小可以简化为

   $$ padding = \frac{(input_{size} - 1) * strides + kernel_{size} - input_{size}}{2} = \frac{kernel_{size} - 1}{2}$$
  
  在我们使用的卷积核尺寸是奇数的情况下，这一步的padding尺寸可以简单的表述为`padding = kernelSize // 2`。

  接下来，我们参考caffe中img2col的做法

  <img src="./Establish/img2col.png" width="600">

  上图表示的算法就是将输入数据按照计算的步骤展开，构成一个尺寸更大的矩阵，同时将每个通道的卷积核展开成一个大的矩阵，这样通过一次矩阵运算，就可以得出卷积层的输出值。

  仍然使用上面单一通道的卷积例子来进行说明，为了表示出展开顺序，卷积核采用
  
  $$
  \begin{pmatrix}
  1 & 2 & 3\\
  4 & 5 & 6\\
  7 & 8 & 9
  \end{pmatrix}
  $$

  我们经过padding之后的输入是

  <img src="./Establish/padding.png" width="300">

  卷积核尺寸是3*3，我们把输入进行展开，将卷积运算一次的内容放入一行中，并把每个卷积核按行展开成一列，
  得到的用于运算的输入是：

  $$
  \begin{pmatrix}
   0& 0& 0& 0& 10& 8& 0& 2& 2\\
   0& 0& 0& 10& 8& 9& 2& 2& 12\\
   0& 0& 0& 8& 9& 7& 2& 12& 3\\
   0& 0& 0& 9& 7& 0& 12& 3& 0\\
   \vdots & \vdots &\vdots &\vdots &\vdots &\vdots &\vdots &\vdots &\vdots &\\
   8 & 9 & 0 & 23 & 12 & 0 & 0 & 0 & 0
  \end{pmatrix}
  $$

  用于运算的卷积核展开结果是

  $$
  \begin{pmatrix}
  1 \\
  2 \\
  3 \\
  4 \\
  5 \\
  6 \\
  7 \\
  8 \\
  9 \\
  \end{pmatrix}
  $$

  对于多通道输入，比如对一个两个通道的，内容完全一致的上述矩阵进行运算，卷积核输出也为两个通道，我们可以简单的从上面的过程中得到扩展：

  对于输入，会被展开成

  $$
  \begin{pmatrix}
   0& 0& 0& 0& 10& 8& 0& 2& 2 & 0& 0& 0& 0& 10& 8& 0& 2& 2\\
   0& 0& 0& 10& 8& 9& 2& 2& 12 & 0& 0& 0& 10& 8& 9& 2& 2& 12\\
   0& 0& 0& 8& 9& 7& 2& 12& 3 & 0& 0& 0& 8& 9& 7& 2& 12& 3\\
   0& 0& 0& 9& 7& 0& 12& 3& 0 & 0& 0& 0& 9& 7& 0& 12& 3& 0\\
   \vdots & \vdots &\vdots &\vdots &\vdots &\vdots &\vdots &\vdots &\vdots &\vdots & \vdots &\vdots &\vdots &\vdots &\vdots &\vdots &\vdots &\vdots\\
   8 & 9 & 0 & 23 & 12 & 0 & 0 & 0 & 0 & 8 & 9 & 0 & 23 & 12 & 0 & 0 & 0 & 0
  \end{pmatrix}
  $$

  也就是简单的将第二个通道的内容展开到第一个通道内容的后面。对于卷积核做类似的操作，但是不同输出通道的卷积核需要被放置在不同的列，每一列对应一个卷积核。

  $$
  \begin{pmatrix}
  1 & 1\\
  2 & 2\\
  3 & 3\\
  4 & 4\\
  5 & 5\\
  6 & 6\\
  7 & 7\\
  8 & 8\\
  9 & 9\\
  1 & 1\\
  2 & 2\\
  3 & 3\\
  4 & 4\\
  5 & 5\\
  6 & 6\\
  7 & 7\\
  8 & 8\\
  9 & 9
  \end{pmatrix}
  $$

  通过img2col的操作，我们可以通过一次矩阵乘法就得到所有位置的卷积结果，再通过一次reshape操作，将我们的输出数据还原到期望的输出格式，我们的前向传播也就完成啦！

+ 反向传播：  
  不同于之前所说的max_pool和dropout，卷积层自己是有卷积核这样一个参数的，所以在卷积层的反向传播，既要将本层的梯度回传，又要利用传递来的参数来更新本层自己的参数值。

  从简单的入手吧。先来看如何利用传递回来的参数来更新自己的参数值吧。

  先来看反向传播的四大公式：
  
  $$\delta^{L} = \nabla_{a}C \odot \sigma_{'}(Z^L) \tag{1}$$
  $$\delta^{l} = ((W^{l + 1})^T\delta^{l+1})\odot\sigma_{'}(Z^l) \tag{2}$$
  $$\frac{\partial{C}}{\partial{b_j^l}} = \delta_j^l \tag{3}$$
  $$\frac{\partial{C}}{\partial{w_{jk}^{l}}} = a_k^{l-1}\delta_j^l \tag{4}$$

  在这里，我们逐层传递的就是$\delta^{l}$这一项。根据传递回来的$\delta^{l + 1}$，我们可以分别计算本层偏置和权重参数对应的梯度和需要继续传播的$\delta^{l}$。
  
  然而，卷积层的传播公式和这四大公式有些许不同，因为卷积层的参数是共享的。但是其中的原理是相同的。下面来具体探究下在卷积层运算中需要经过怎样的改变来调整卷积层的传播公式。
  <!-- 以一个单通道的传递为例来描述这个反向传播的过程。 -->

  在梯度计算中，本层的和要计算梯度的参数相关的数据是与计算这个参数的梯度紧密相关的，这是来源于梯度反向传播链式法则中的
  
  $$\frac{\partial{C}}{\partial{w_{jk}^{l}}} = \frac{\partial{z^l}}{\partial{w_{jk}^{l}}}\delta_j^l$$

  在上式中，$z_l$是当前层(第l层）的输出。在卷积层中，和$w_{j,k}$有关的变量有多少呢？

  这就要再次涉及到我们之前使用的img2col方法了。比如在我们上面的例子中，和最左上角的1有关的输入中的数据是哪一些呢？根据矩阵乘法的运算法则，我们会很容易发现，和这个1有关的数据都在我们展开的输入数据的第一行。以此类推，我们可以轻松的找出和每个卷积核参数有关系的输入数据。那么，在这里，我们考虑展开后的卷积核，会有
  
  $$\frac{\partial{C}}{\partial{w_{j}^{l}}} = \frac{\partial{z^l}}{\partial{w_{j}^{l}}}\delta_j^l = \sum_{i}x_{j,i}\delta_{i}^{l}$$

  $x_{j,i}$表示展开后的矩阵的第j行第i列的数据。在这里，这样一个有规律的求和形式可以用什么来代替呢？矩阵乘法！我们将反向传播得到的$\delta$矩阵进行按行进行展开，并且img2col的展开矩阵结果进行转置，这样，通过一次矩阵乘法，我们就可以得到所有的梯度值。用上面单通道的例子来说明这个问题，假设我们得到的$\delta$矩阵是

  $$
  \begin{pmatrix}
  1 & 2& 3& 4& 5\\
  6 & 7& 8& 9& 10\\
  11& 12& 13& 14& 15\\
  16& 17& 18& 19& 20\\
  21& 22& 23& 24& 25
  \end{pmatrix}
  $$

  那么我们将上面的矩阵按行顺序展开，得到的结果是

  $$
  \begin{pmatrix}
  1 \\
  2 \\
  3 \\
  \vdots\\
  25
  \end{pmatrix}
  $$

  将img2col得到的展开输入矩阵转置，并于上面的展开矩阵相乘，会发现，得到的结果矩阵中的每一项都是我们所期望得到的梯度项啦！

  现在我们得到了本层参数更新的梯度，下面就是反向传播的另一个方面，关于如何计算本层的$\delta$。

  根据反向传播的公式2
  
  $$\delta^{l - 1} = ((W^{l})^T\delta^{l})\odot\sigma_{'}(Z^{l - 1})$$
  
  因为我们将激活函数和卷积层进行了分开的处理，所以这里的$\sigma_{'}(z^l)$恒定为1，因此我们的重点就是寻找本层的$w$参数和$\delta^{l}$的关系。这里我们考虑输入中的一个元素对应的输出中的元素，这样的一个输入元素会依次被覆盖多次

  <img src=".\Establish\convback.png" width="600">

  从上面的卷积示意图中可以看到，对于在A附近的一个3\*3的输出，在反向传播过程中，左上角的输出$z_1$对应的是A元素和卷积核$w_{3,3}$，$z_2$对应的是A元素和卷积核$w_{3,2}$，以此类推，不难发现这个顺序就是对卷积核的转置。那么，用一个转置的$w$矩阵和回传的$\delta$矩阵在$z_1$到$z_9$的位置做卷积操作就可以得到位置A处的$\delta$值。对其他元素也不难发现类似的规律。也就是说，将公式二中的矩阵乘法替换成卷积就可以得到反向传播的结果。

  所以，像这样的权重等于是将卷积核做转置之后与传递回来的$\delta^{l}$做卷积操作。那么，我们只需要将前向传播中的img2col用到$\delta^{l}$，并将结果和$W^T$作卷积操作就可得到可以继续传播的$\delta^{l - 1}$了。

+ 示例代码：

  ```python
  class Cconv2d(object):
    def __init__(self, inputSize, kernelSize, outputChannel, stride=1, padding="SAME"):

        # 初始化的过程

        self.inputSize = inputSize
        self.kernelSize = kernelSize
        self.stride = stride
        self.batch = self.inputSize[0]
        self.inputChannel = inputSize[-1]
        self.outputChannel = outputChannel
        self.padding = padding

        # 生成初始化权重

        weights_scale = math.sqrt(reduce(lambda x, y: x * y, inputSize)) / outputChannel

        self.weights = np.random.standard_normal((kernelSize, kernelSize, self.inputChannel, outputChannel)) / weights_scale
        self.bias = np.random.standard_normal(outputChannel) / weights_scale
        self.weightsGrad = np.zeros(self.weights.shape)
        self.biasGrad = np.zeros(self.bias.shape)
        self.backError = np.zeros((inputSize[0], inputSize[1] // stride, inputSize[2] // stride, outputChannel))
        self.outputShape = self.backError.shape

    # img2col的展开过程，将图片根据卷积核尺寸和步长等进行展开

    def expand(self, image, kernelSize, stride):

        colImage = []
        for i in range(0, image.shape[1] - kernelSize + 1, stride):
            for j in range(0, image.shape[2] - kernelSize + 1, stride):
                col = image[:, i:i + kernelSize, j:j+kernelSize, :].reshape(-1)
                colImage.append(col)

        return np.array(colImage)

    def forward(self, image):

        weights = np.reshape(self.weights, [-1, self.outputChannel])
        self.image = image
        shape = image.shape

        # 将图像进行加padding的步骤，保证输入输出尺寸一致

        image = np.pad(image,
        ((0, 0), (self.kernelSize // 2, self.kernelSize // 2), (self.kernelSize // 2, self.kernelSize // 2), (0, 0)),
        mode='constant', constant_values=0)

        self.colImage = []
        convOut = np.zeros(self.backError.shape)
        for i in range(shape[0]):

            # 计算卷积

            colImage = self.expand(image[i][np.newaxis, :], self.kernelSize, self.stride)
            convOut[i] = np.reshape(np.dot(colImage, weights) + self.bias, self.backError[0].shape)
            self.colImage.append(colImage)

        # 记录展开后的输入数据

        self.colImage = np.array(self.colImage)

        return convOut

    def gradient(self, preError):

        self.backError = preError
        preError = np.reshape(preError, [self.inputSize[0], -1, self.outputChannel])

        # 根据回传的误差计算本层的梯度

        for i in range(self.inputSize[0]):
            self.weightsGrad = self.weightsGrad + np.dot(self.colImage[i].T, preError[i]).reshape(self.weights.shape)
        self.biasGrad = self.biasGrad + np.sum(preError, (0, 1))

        # 将回传的误差加上padding，按照推导过程，用卷积计算回传的误差

        preError = np.pad(self.backError,
        ((0, 0), (self.kernelSize // 2, self.kernelSize //2), (self.kernelSize // 2, self.kernelSize // 2), (0, 0)),
        mode='constant', constant_values=0)

        # 将卷积核转置，并进行展开，注意输入输出通道数需要交换

        weights = np.flipud(np.fliplr(self.weights))
        weights = weights.swapaxes(2, 3)
        weights = np.reshape(weights, [-1, self.inputChannel])
        backError = np.zeros(self.inputSize)

        # 用卷积的形式继续计算误差

        for i in range(self.inputSize[0]):
            backError[i] = np.reshape(np.dot(self.expand(preError[i][np.newaxis, :], self.kernelSize, self.stride), weights), self.inputSize[1:4])

        return backError

    def backward(self, learningRate=0.0001, weightsDecay=0.0004):

        # 根据梯度进行更新，此处添加了动量的因素

        weights = (1 - weightsDecay) * self.weights
        bias = (1 - weightsDecay) * self.bias
        self.weights = weights - learningRate * self.weightsGrad
        self.bias = bias - learningRate * self.biasGrad

        # 添加了动量而不是单纯的置零

        self.weightsGrad = 0.9 * self.weightsGrad
        self.biasGrad = 0.9 * self.biasGrad
  ```


### `softmax` & `loss`

+ 作用：

  softmax函数和loss函数我们放在一起来进行说明，在tensorflow有关函数的接口中，函数体内部会自动执行一个softmax操作。什么是softmax操作呢？

  从softmax的函数形式来入手。

  $$ P_i = \frac{e^{{z_i}}}{\sum\limits_j{e^{z_j}}}$$

  这个函数的作用是是什么呢？从直观上来说，就是将一堆的0，1，2，3这样的数，转化成一群0~1之间，并且总和为1的概率分布。之后，我们可以选择概率最大的那一个选项，作为我们的输出。之后，我们就是用这样的概率分布和预期的概率分布去计算loss。在这里，我们采用交叉熵作为损失函数，由于是一个只有一类是正确的问题，所以目标的概率分布都是只有一项是1其余项均是零这样一种形式，所以，loss函数可以简单的写成：

  $$loss = -log (P_{choose}) = -log(softmax(z_j))$$

  上式中，$j$代表了被选中的那一项。用$i$代表正确的选项。如果$i=j$，将$loss$对$z_i$求出梯度：

  $$\frac{\partial{loss}}{\partial{z_i}} = -\frac{\partial{log(softmax(z_i))}}{\partial{z_i}} = \frac{\sum\limits_j{e^{z_j}}}{e^{z_i}}*\frac{e^{z_i} - \sum\limits_j{e^{z_j}}}{(\sum\limits_j{e^{z_j}})^2}*e^{z_i} = \frac{e^{{z_i}}}{\sum\limits_j{e^{z_j}}} - 1 = P_i - 1$$

  对于$i \neq j$的情况，将$loss$对$z_i$求出梯度：
  
  $$\frac{\partial{loss}}{\partial{z_i}} = -\frac{\partial{log(softmax(z_j))}}{\partial{z_i}} = \frac{\sum\limits_j{e^{z_j}}}{e^{z_j}}*\frac{e^{z_j}}{(\sum\limits_j{e^{z_j}})^2}*e^{z_i} = \frac{e^{{z_i}}}{\sum\limits_j{e^{z_j}}} = P_i$$

  所以，我们可以得出根据这样的函数形式和反向传播写出示例代码如下：

  ```python
  class Csoftmax(object):
    def __init__(self, inputSize):
        self.shape = inputSize
        self.error = np.zeros(inputSize)
        self.batchSize = inputSize[0]

    def calLoss(self, labels, perdiction):
        self.label = labels
        self.softmax = np.zeros(self.shape)
        self.loss = 0

        for i in range(self.batchSize):
            perdiction[i] = perdiction[i] - np.max(perdiction[i])

            # 计算softmax的概率

            perdiction[i] = np.exp(perdiction[i])
            self.softmax[i] = perdiction[i] / np.sum(perdiction[i])

            # 根据label和预测结果，按公式计算损失大小

            self.loss = self.loss - np.log(self.softmax[i, labels[i]])

        return self.loss

    def gradient(self):

        # 按照公式，计算回传的梯度大小

        self.error = self.softmax.copy()
        for i in range(self.batchSize):
            self.error[i, self.label[i]] -= 1

        return self.error
  ```

  参考文章:

  [https://zhuanlan.zhihu.com/c_162633442](https://zhuanlan.zhihu.com/c_162633442)