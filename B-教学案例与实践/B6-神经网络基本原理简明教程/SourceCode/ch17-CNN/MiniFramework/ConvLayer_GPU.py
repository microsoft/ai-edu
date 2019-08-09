# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from MiniFramework.EnumDef_6_0 import *
from MiniFramework.Layer import *
from MiniFramework.ConvKernal import *

class ConvLayer_GPU(CLayer):
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
        (self.output_height, self.output_width) = calculate_output_size(
            self.input_height, self.input_width, 
            self.filter_height, self.filter_width, 
            self.padding, self.stride)
        self.output_shape = (self.num_output_channel, self.output_height, self.output_height)

    def forward(self, x, train=True):
        self.x = x
        assert(self.x.shape[1] == self.num_input_channel)
        assert(self.x.shape[2] == self.input_height)
        assert(self.x.shape[3] == self.input_width)
        self.batch_size = self.x.shape[0]
        FN, C, FH, FW = self.Kernal.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.padding - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.padding - FW) / self.stride)
        self.col_x = im2col(x, FH, FW, self.stride, self.padding)
        self.col_w = self.Kernal.W.reshape(FN, -1).T
        out = np.dot(self.col_x, self.col_w) + self.Kernal.B.reshape(-1,FN)
        self.z = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) 
        return self.z

    def backward(self, delta_in, layer_idx):
        FN, C, FH, FW = self.Kernal.W.shape
        dout = delta_in.transpose(0,2,3,1).reshape(-1, FN)
        self.Kernal.dB = np.sum(dout, axis=0, keepdims=True).T / self.batch_size
        dW = np.dot(self.col_x.T, dout)
        self.Kernal.dW = dW.transpose(1, 0).reshape(FN, C, FH, FW) / self.batch_size
        dcol = np.dot(dout, self.col_w.T)
        delta_out = col2im(dcol, self.x.shape, FH, FW, self.stride, self.padding)
        return delta_out
    
    def backward_gpu(self, delta_in, layer_idx):
        delta_out = self.backward(delta_in, layer_idx)
        return delta_out, self.Kernal.dW, self.Kernal.dB

    def pre_update(self):
        self.weights.pre_Update()

    def update(self):
        self.Kernal.Update()
        
    def save_parameters(self):
        self.Kernal.SaveResultValue()

    def load_parameters(self):
        self.Kernal.LoadResultValue()

#end class

def calculate_output_size(input_h, input_w, filter_h, filter_w, padding, stride=1):
    output_h = (input_h - filter_h + 2 * padding) // stride + 1    
    output_w = (input_w - filter_w + 2 * padding) // stride + 1
    return (output_h, output_w)

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w)).astype(np.float32)
    #col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for i in range(filter_h):
        i_max = i + stride*out_h
        for j in range(filter_w):
            j_max = j + stride*out_w
            col[:, :, i, j, :, :] = img[:, :, i:i_max:stride, j:j_max:stride]
        #end for
    #end for
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1)).astype(np.float32)
    #img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for i in range(filter_h):
        i_max = i + stride*out_h
        for j in range(filter_w):
            j_max = j + stride*out_w
            img[:, :, i:i_max:stride, j:j_max:stride] += col[:, :, i, j, :, :]
        #end for
    #end for
    return img[:, :, pad:H + pad, pad:W + pad]

from MiniFramework.HyperParameters_4_2 import *
import time

if __name__ == '__main__':
    
    batch_size = 32

    params = HyperParameters_4_2(
        0.1, 1, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier)

    stride = 1
    padding = 1
    fh = 3
    fw = 3
    input_channel = 1
    output_channel = 8
    iw = 28
    ih = 28

    # 64 个 3 x 28 x 28 的图像输入（模拟 mnist）
    c1 = ConvLayer_GPU((input_channel,iw,ih), (output_channel,fh,fw), (stride, padding), params)
    c1.initialize("test", "test", False)
    x = np.random.randn(batch_size, input_channel, iw, ih)
    f1 = c1.forward_numba(x)
    delta_in = np.ones((f1.shape))
    b1, dw1, db1 = c1.backward_numba(delta_in, 1)

    c2 = ConvLayer_GPU((input_channel,iw,ih), (output_channel,fh,fw), (stride, padding), params)
    c2.initialize("test", "test", False)
    f2 = c2.forward(x)
    delta_in = np.ones((f2.shape))
    b2, dw2, db2 = c2.backward_gpu(delta_in, 1)
    
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
    print(db1)
    print(db2)


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
