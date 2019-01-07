# codes to transfer a model written in numpy to onnx

目录下的文件说明:

- `xxx.npy`： 用于存储训练后的数据和变量
- `mnist_np.py`: 使用numpy训练网络
- `transfer.py`: 将numpy的model转换成onnx model
- `save.py`: 将numpy模型保存为json+对应数据

运行步骤：

1. `mnist_np.py` 存储得到的numpy数据
2. `transfer.py` 将模型转变为onnx
3. 将模型拷贝到适当的位置进行下一步操作
