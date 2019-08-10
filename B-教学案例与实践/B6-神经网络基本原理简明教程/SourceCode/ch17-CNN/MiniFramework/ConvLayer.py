# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from MiniFramework.EnumDef_6_0 import *
from MiniFramework.Layer import *
from MiniFramework.ConvKernal import *
from MiniFramework.utility import *
from MiniFramework.jit_utility import *

class ConvLayer(CLayer):
    # define the number of input and output channel, also the filter size
    def __init__(self, 
                 input_shape,       # (InputChannelCount, H, W)
                 kernal_shape,      # (OutputChannelCount, FH, FW)
                 conv_param,        # (stride, padding)
                 hp):
        self.num_input_channel = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.num_output_channel = kernal_shape[0]
        self.filter_height = kernal_shape[1]
        self.filter_width = kernal_shape[2]
        self.stride = conv_param[0]
        self.padding = conv_param[1]
        self.hp = hp

    def initialize(self, folder, name, create_new=False):
        self.Kernal = ConvKernal(
            self.num_output_channel, self.num_input_channel, self.filter_height, self.filter_width, 
            self.hp.init_method, self.hp.optimizer_name, self.hp.eta)
        self.Kernal.Initialize(folder, name, create_new)
        (self.output_height, self.output_width) = ConvLayer.calculate_output_size(
            self.input_height, self.input_width, 
            self.filter_height, self.filter_width, 
            self.padding, self.stride)
        self.output_shape = (self.num_output_channel, self.output_height, self.output_height)

    def forward(self, x, train=True):
        return self.forward_img2col(x, train)

    def backward(self, delta_in, layer_idx):
        delta_out, dw, db = self.backward_col2img(delta_in, layer_idx)
        return delta_out

    def forward_img2col(self, x, train=True):
        self.x = x
        assert(self.x.shape[1] == self.num_input_channel)
        assert(self.x.shape[2] == self.input_height)
        assert(self.x.shape[3] == self.input_width)
        self.batch_size = self.x.shape[0]
        FN, C, FH, FW = self.Kernal.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.padding - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.padding - FW) / self.stride)
        self.col_x = img2col(x, FH, FW, self.stride, self.padding)
        self.col_w = self.Kernal.W.reshape(FN, -1).T
        out1 = np.dot(self.col_x, self.col_w) + self.Kernal.B.reshape(-1,FN)
        out2 = out1.reshape(N, out_h, out_w, -1)
        self.z = np.transpose(out2, axes=(0, 3, 1, 2))
        return self.z

    def backward_col2img(self, delta_in, layer_idx):
        FN, C, FH, FW = self.Kernal.W.shape
        dout = np.transpose(delta_in, axes=(0,2,3,1)).reshape(-1, FN)
        self.Kernal.dB = np.sum(dout, axis=0, keepdims=True).T / self.batch_size
        dW = np.dot(self.col_x.T, dout)
        self.Kernal.dW = np.transpose(dW, axes=(1, 0)).reshape(FN, C, FH, FW) / self.batch_size
        dcol = np.dot(dout, self.col_w.T)
        delta_out = col2img(dcol, self.x.shape, FH, FW, self.stride, self.padding)
        return delta_out, self.Kernal.dW, self.Kernal.dB
   
    
    def forward_numba(self, x, train=True):
        assert(x.ndim == 4)
        self.x = x
        assert(self.x.shape[1] == self.num_input_channel)
        assert(self.x.shape[2] == self.input_height)
        assert(self.x.shape[3] == self.input_width)
        self.batch_size = self.x.shape[0]

        if self.padding > 0:
            self.padded = np.pad(self.x, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), 'constant')
            #self.padded = np.pad(self.x, mode="constant", constant_value=0, pad_width=(0,0,0,0,self.padding,self.padding,self.padding,self.padding))
        else:
            self.padded = self.x
        #end if

        self.z = jit_conv_4d(self.padded, self.Kernal.W, self.Kernal.B, self.output_height, self.output_width, self.stride)
        return self.z

    def backward_numba(self, delta_in, flag):
        assert(delta_in.ndim == 4)
        assert(delta_in.shape == self.z.shape)
        
        # 转换误差矩阵尺寸
        dz_stride_1 = expand_delta_map(delta_in, self.batch_size, self.num_output_channel, self.input_height, self.input_width, self.output_height, self.output_width, self.filter_height, self.filter_width, self.padding, self.stride)

        # 求本层的输出误差矩阵时，应该用本层的输入误差矩阵互相关计算本层的卷积核的旋转
        # 由于输出误差矩阵的尺寸必须与本层的输入数据的尺寸一致，所以必须根据卷积核的尺寸，调整本层的输入误差矩阵的尺寸
        (pad_h, pad_w) = calculate_padding_size(
            dz_stride_1.shape[2], dz_stride_1.shape[3], 
            self.filter_height, self.filter_width, 
            self.input_height, self.input_width)
        
        dz_padded = np.pad(dz_stride_1, ((0,0),(0,0),(pad_h, pad_h),(pad_w, pad_w)), 'constant')

        # 计算本层的权重矩阵的梯度
        self._calculate_weightsbias_grad(dz_stride_1)

        # 计算本层输出到下一层的误差矩阵
        delta_out = self._calculate_delta_out(dz_padded, flag)
        #return delta_out
        return delta_out, self.Kernal.dW, self.Kernal.dB

    # 用输入数据乘以回传入的误差矩阵,得到卷积核的梯度矩阵
    def _calculate_weightsbias_grad(self, dz):
        self.Kernal.ClearGrads()
        # 先把输入矩阵扩大，周边加0
        (pad_h, pad_w) = calculate_padding_size(
            self.input_height, self.input_width, 
            dz.shape[2], dz.shape[3], 
            self.filter_height, self.filter_width, 1)
        input_padded = np.pad(self.x, ((0,0),(0,0),(pad_h, pad_h),(pad_w,pad_w)), 'constant')
        # 输入矩阵与误差矩阵卷积得到权重梯度矩阵
        (self.Kernal.dW, self.Kernal.dB) = calcalate_weights_grad(
                                input_padded, dz, self.batch_size, 
                                self.num_output_channel, self.num_input_channel, 
                                self.filter_height, self.filter_width, 
                                self.Kernal.dW, self.Kernal.dB)

        self.Kernal.MeanGrads(self.batch_size)

        
    # 用输入误差矩阵乘以（旋转180度后的）卷积核
    def _calculate_delta_out(self, dz, layer_idx):
        if layer_idx == 0:
            return None
        # 旋转卷积核180度
        rot_weights = self.Kernal.Rotate180()
        delta_out = np.zeros(self.x.shape).astype(np.float32)
        # 输入梯度矩阵卷积旋转后的卷积核，得到输出梯度矩阵
        delta_out = calculate_delta_out(dz, rot_weights, self.batch_size, 
                            self.num_input_channel, self.num_output_channel, 
                            self.input_height, self.input_width, delta_out)

        return delta_out

    def pre_update(self):
        self.weights.pre_Update()

    def update(self):
        self.Kernal.Update()
        
    def save_parameters(self):
        self.Kernal.SaveResultValue()

    def load_parameters(self):
        self.Kernal.LoadResultValue()

    @staticmethod
    def calculate_output_size(input_h, input_w, filter_h, filter_w, padding, stride=1):
        output_h = (input_h - filter_h + 2 * padding) // stride + 1    
        output_w = (input_w - filter_w + 2 * padding) // stride + 1
        return (output_h, output_w)

#end class

from MiniFramework.HyperParameters_4_2 import *
from MiniFramework.ConvLayer import *

import time

if __name__ == '__main__':
    
    batch_size = 64

    params = HyperParameters_4_2(
        0.1, 1, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier)

    stride = 1
    padding = 1
    fh = 3
    fw = 3
    input_channel = 3
    output_channel = 4
    iw = 24
    ih = 28
    
    x = np.random.randn(batch_size, input_channel, iw, ih)

    # 64 个 3 x 28 x 28 的图像输入（模拟 mnist）
    
    c1 = ConvLayer((input_channel,iw,ih), (output_channel,fh,fw), (stride, padding), params)
    c1.initialize("test", "test", False)
    s1 = time.time()
    for i in range(100):
        f1 = c1.forward2(x)
        delta_in = np.ones((f1.shape))
        b1, dw1, db1 = c1.backward2(delta_in, 1)
    e1 = time.time()
    print(e1-s1)
    
    s2 = time.time()
    for i in range(100):
        f2 = c1.forward(x)
        b2, dw2, db2 = c1.backward_test(delta_in, 1)
    e2 = time.time()
    print(e2-s2)
    
    print(np.allclose(f1, f2, atol=1e-5))
    #print(f1)
    #print(f2)

    print(np.allclose(b1, b2, atol=1e-4))
    #print(b1)
    #print(b2)

    print(np.allclose(dw1, dw2, atol=1e-4))
    #print(dw1)
    #print(dw2)

    print(np.allclose(db1, db2, atol=1e-4))


    exit()

    s = time.time()
    for i in range(10):
        r1 = conv1()
    e1 = time.time()
    print("numba:", e1 - s)

    for i in range(10):
        r2 = conv2()
    e2 = time.time()
    print("im2col:", e2 - e1)
    print(np.allclose(r1, r2, atol=1e-5))
    print(np.allclose(r1, r2, rtol=1e-3))
