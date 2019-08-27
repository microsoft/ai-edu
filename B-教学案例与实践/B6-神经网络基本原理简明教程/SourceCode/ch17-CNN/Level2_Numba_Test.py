# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy
import numba
import time

from MiniFramework.ConvWeightsBias import *
from MiniFramework.ConvLayer import *

def conv_4d(x, weights, bias, out_h, out_w, stride=1):
    # 输入图片的批大小，通道数，高，宽
    assert(x.ndim == 4)
    # 输入图片的通道数
    assert(x.shape[1] == weights.shape[1])  
    batch_size = x.shape[0]
    num_input_channel = x.shape[1]
    num_output_channel = weights.shape[0]
    filter_height = weights.shape[2]
    filter_width = weights.shape[3]
    rs = np.zeros((batch_size, num_output_channel, out_h, out_w))

    for bs in range(batch_size):
        for oc in range(num_output_channel):
            rs[bs,oc] += bias[oc]
            for ic in range(num_input_channel):
                for i in range(out_h):
                    for j in range(out_w):
                        ii = i * stride
                        jj = j * stride
                        for fh in range(filter_height):
                            for fw in range(filter_width):
                                rs[bs,oc,i,j] += x[bs,ic,fh+ii,fw+jj] * weights[oc,ic,fh,fw]
    return rs

@nb.jit(nopython=True)
def jit_conv_4d(x, weights, bias, out_h, out_w, stride=1):
    # 输入图片的批大小，通道数，高，宽
    assert(x.ndim == 4)
    # 输入图片的通道数
    assert(x.shape[1] == weights.shape[1])  
    batch_size = x.shape[0]
    num_input_channel = x.shape[1]
    num_output_channel = weights.shape[0]
    filter_height = weights.shape[2]
    filter_width = weights.shape[3]
    rs = np.zeros((batch_size, num_output_channel, out_h, out_w))

    for bs in range(batch_size):
        for oc in range(num_output_channel):
            rs[bs,oc] += bias[oc]
            for ic in range(num_input_channel):
                for i in range(out_h):
                    for j in range(out_w):
                        ii = i * stride
                        jj = j * stride
                        for fh in range(filter_height):
                            for fw in range(filter_width):
                                rs[bs,oc,i,j] += x[bs,ic,ii+fh,jj+fw] * weights[oc,ic,fh,fw]
    return rs

def calculate_output_size(input_h, input_w, filter_h, filter_w, padding, stride=1):
    output_h = (input_h - filter_h + 2 * padding) // stride + 1    
    output_w = (input_w - filter_w + 2 * padding) // stride + 1
    return (output_h, output_w)

if __name__ == '__main__':
    stride = 1
    padding = 0
    fh = 3
    fw = 3
    input_channel = 3
    output_channel = 4
    iw = 28
    ih = 28
    (output_height, output_width) = calculate_output_size(ih, iw, fh, fw, padding, stride)
    wb = ConvWeightsBias(output_channel, input_channel, fh, fw, InitialMethod.MSRA, OptimizerName.SGD, 0.1)
    wb.Initialize("test", "test", True)
    batch_size = 64
    x = np.random.randn(batch_size, input_channel, iw, ih)
    # dry run
    output1 = conv_4d(x, wb.W, wb.B, output_height, output_width, stride)
    s1 = time.time()
    for i in range(10):
        output1 = conv_4d(x, wb.W, wb.B, output_height, output_width, stride)
    e1 = time.time()
    print("Time used for Python:", e1 - s1)

    # dry run
    output2 = jit_conv_4d(x, wb.W, wb.B, output_height, output_width, stride)
    s2 = time.time()
    for i in range(10):
        output2 = jit_conv_4d(x, wb.W, wb.B, output_height, output_width, stride)
    e2 = time.time()
    print("Time used for Numba:", e2 - s2)

    print("correctness:", np.allclose(output1, output2, atol=1e-7))
