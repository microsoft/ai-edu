# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from MiniFramework.EnumDef_6_0 import *
from MiniFramework.Layer import *
from MiniFramework.ConvWeightsBias import *
from MiniFramework.jit_utility import *

class ConvLayer(CLayer):
    # define the number of input and output channel, also the filter size
    def __init__(self, 
                 input_shape,       # (InputChannelCount, H, W)
                 kernal_shape,      # (OutputChannelCount, FH, FW)
                 conv_param,        # (stride, padding)
                 hp):
        self.InC = input_shape[0]   # input channel count
        self.InH = input_shape[1]   # input image height
        self.InW = input_shape[2]   # input image width
        self.OutC = kernal_shape[0] # output channel count
        self.FH = kernal_shape[1]   # kernal/filter height
        self.FW = kernal_shape[2]   # kernal/filter width
        self.stride = conv_param[0]
        self.padding = conv_param[1]
        self.hp = hp

    def initialize(self, folder, name, create_new=False):
        self.WB = ConvWeightsBias(
            self.OutC, self.InC, self.FH, self.FW, 
            self.hp.init_method, self.hp.optimizer_name, self.hp.eta)
        self.WB.Initialize(folder, name, create_new)
        (self.OutH, self.OutW) = calculate_output_size(
            self.InH, self.InW, 
            self.FH, self.FW, 
            self.padding, self.stride)
        self.output_shape = (self.OutC, self.OutH, self.OutH)

    def set_filter(self, w, b):
        if w is not None:
            self.WB.W = w
        if b is not None:
            self.WB.B = b

    def forward(self, x, train=True):
        return self.forward_img2col(x, train)

    def backward(self, delta_in, layer_idx):
        delta_out, dw, db = self.backward_col2img(delta_in, layer_idx)
        return delta_out

    def forward_img2col(self, x, train=True):
        self.x = x
        self.batch_size = self.x.shape[0]
        assert(self.x.shape == (self.batch_size, self.InC, self.InH, self.InW))
        self.col_x = img2col(x, self.FH, self.FW, self.stride, self.padding)
        self.col_w = self.WB.W.reshape(self.OutC, -1).T
        self.col_b = self.WB.B.reshape(-1, self.OutC)
        out1 = np.dot(self.col_x, self.col_w) + self.col_b
        out2 = out1.reshape(self.batch_size, self.OutH, self.OutW, -1)
        self.z = np.transpose(out2, axes=(0, 3, 1, 2))
        return self.z

    def backward_col2img(self, delta_in, layer_idx):
        col_delta_in = np.transpose(delta_in, axes=(0,2,3,1)).reshape(-1, self.OutC)
        self.WB.dB = np.sum(col_delta_in, axis=0, keepdims=True).T / self.batch_size
        col_dW = np.dot(self.col_x.T, col_delta_in) / self.batch_size
        self.WB.dW = np.transpose(col_dW, axes=(1, 0)).reshape(self.OutC, self.InC, self.FH, self.FW)
        col_delta_out = np.dot(col_delta_in, self.col_w.T)
        delta_out = col2img(col_delta_out, self.x.shape, self.FH, self.FW, self.stride, self.padding, self.OutH, self.OutW)
        return delta_out, self.WB.dW, self.WB.dB
   
    def forward_numba(self, x, train=True):
        assert(x.ndim == 4)
        self.x = x
        assert(self.x.shape[1] == self.InC)
        assert(self.x.shape[2] == self.InH)
        assert(self.x.shape[3] == self.InW)
        self.batch_size = self.x.shape[0]

        if self.padding > 0:
            self.padded = np.pad(self.x, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), 'constant')
            #self.padded = np.pad(self.x, mode="constant", constant_value=0, pad_width=(0,0,0,0,self.padding,self.padding,self.padding,self.padding))
        else:
            self.padded = self.x
        #end if

        self.z = jit_conv_4d(self.padded, self.WB.W, self.WB.B, self.OutH, self.OutW, self.stride)
        return self.z

    def backward_numba(self, delta_in, flag):
        assert(delta_in.ndim == 4)
        assert(delta_in.shape == self.z.shape)
        
        # 如果正向计算中的stride不是1，转换成是1的等价误差数组
        dz_stride_1 = expand_delta_map(delta_in, self.batch_size, self.OutC, self.InH, self.InW, self.OutH, self.OutW, self.FH, self.FW, self.padding, self.stride)

        # 计算本层的权重矩阵的梯度
        self._calculate_weightsbias_grad(dz_stride_1)

        # 求本层的输出误差矩阵时，应该用本层的输入误差矩阵互相关计算本层的卷积核的旋转
        # 由于输出误差矩阵的尺寸必须与本层的输入数据的尺寸一致，所以必须根据卷积核的尺寸，调整本层的输入误差矩阵的尺寸
        (pad_h, pad_w) = calculate_padding_size(
            dz_stride_1.shape[2], dz_stride_1.shape[3], 
            self.FH, self.FW, 
            self.InH, self.InW)
        dz_padded = np.pad(dz_stride_1, ((0,0),(0,0),(pad_h, pad_h),(pad_w, pad_w)), 'constant')
        # 计算本层输出到下一层的误差矩阵
        delta_out = self._calculate_delta_out(dz_padded, flag)
        #return delta_out
        return delta_out, self.WB.dW, self.WB.dB

    # 用输入数据乘以回传入的误差矩阵,得到卷积核的梯度矩阵
    def _calculate_weightsbias_grad(self, dz):
        self.WB.ClearGrads()
        # 先把输入矩阵扩大，周边加0
        (pad_h, pad_w) = calculate_padding_size(
            self.InH, self.InW, 
            dz.shape[2], dz.shape[3], 
            self.FH, self.FW, 1)
        input_padded = np.pad(self.x, ((0,0),(0,0),(pad_h, pad_h),(pad_w,pad_w)), 'constant')
        # 输入矩阵与误差矩阵卷积得到权重梯度矩阵
        (self.WB.dW, self.WB.dB) = calcalate_weights_grad(
                                input_padded, dz, self.batch_size, 
                                self.OutC, self.InC, 
                                self.FH, self.FW, 
                                self.WB.dW, self.WB.dB)

        self.WB.MeanGrads(self.batch_size)
        
    # 用输入误差矩阵乘以（旋转180度后的）卷积核
    def _calculate_delta_out(self, dz, layer_idx):
        if layer_idx == 0:
            return None
        # 旋转卷积核180度
        rot_weights = self.WB.Rotate180()
        # 定义输出矩阵形状
        delta_out = np.zeros(self.x.shape).astype(np.float32)
        # 输入梯度矩阵卷积旋转后的卷积核，得到输出梯度矩阵
        delta_out = calculate_delta_out(dz, rot_weights, self.batch_size, 
                            self.InC, self.OutC, 
                            self.InH, self.InW, delta_out)

        return delta_out

    def pre_update(self):
        pass

    def update(self):
        self.WB.Update()
        
    def save_parameters(self):
        self.WB.SaveResultValue()

    def load_parameters(self):
        self.WB.LoadResultValue()

#end class
