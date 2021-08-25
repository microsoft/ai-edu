# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy
import numba
import time

from MiniFramework.ConvWeightsBias import *
from MiniFramework.ConvLayer import *
from MiniFramework.HyperParameters_4_2 import *

def calculate_output_size(input_h, input_w, filter_h, filter_w, padding, stride=1):
    output_h = (input_h - filter_h + 2 * padding) // stride + 1    
    output_w = (input_w - filter_w + 2 * padding) // stride + 1
    return (output_h, output_w)

def understand_4d_col2img_simple():
    batch_size = 1
    stride = 1
    padding = 0
    fh = 2
    fw = 2
    input_channel = 1
    output_channel = 1
    iw = 3
    ih = 3
    (output_height, output_width) = calculate_output_size(ih, iw, fh, fw, padding, stride)
    wb = ConvWeightsBias(output_channel, input_channel, fh, fw, InitialMethod.MSRA, OptimizerName.SGD, 0.1)
    wb.Initialize("test", "test", True)
    wb.W = np.array(range(output_channel * input_channel * fh * fw)).reshape(output_channel, input_channel, fh, fw)
    wb.B = np.array([0])
    x = np.array(range(input_channel * iw * ih * batch_size)).reshape(batch_size, input_channel, ih, iw)
    print("x=\n", x)
    col_x = img2col(x, fh, fw, stride, padding)
    print("col_x=\n", col_x)
    print("w=\n", wb.W)
    col_w = wb.W.reshape(output_channel, -1).T
    print("col_w=\n", col_w)

    # backward
    delta_in = np.array(range(batch_size*output_channel*output_height*output_width)).reshape(batch_size, output_channel, output_height, output_width)
    print("delta_in=\n", delta_in)

    delta_in_2d = np.transpose(delta_in, axes=(0,2,3,1)).reshape(-1, output_channel)
    print("delta_in_2d=\n", delta_in_2d)

    dB = np.sum(delta_in_2d, axis=0, keepdims=True).T / batch_size
    print("dB=\n", dB)
    dW = np.dot(col_x.T, delta_in_2d) / batch_size
    print("dW=\n", dW)
    dW = np.transpose(dW, axes=(1, 0)).reshape(output_channel, input_channel, fh, fw)
    print("dW=\n", dW)
    dcol = np.dot(delta_in_2d, col_w.T)
    print("dcol=\n", dcol)
    delta_out = col2img(dcol, x.shape, fh, fw, stride, padding, output_height, output_width)
    print("delta_out=\n", delta_out)


def understand_4d_col2img_complex():
    batch_size = 2
    stride = 1
    padding = 0
    fh = 2
    fw = 2
    input_channel = 3
    output_channel = 2
    iw = 3
    ih = 3
    (output_height, output_width) = calculate_output_size(ih, iw, fh, fw, padding, stride)
    wb = ConvWeightsBias(output_channel, input_channel, fh, fw, InitialMethod.MSRA, OptimizerName.SGD, 0.1)
    wb.Initialize("test", "test", True)
    wb.W = np.array(range(output_channel * input_channel * fh * fw)).reshape(output_channel, input_channel, fh, fw)
    wb.B = np.array([0])
    x = np.array(range(input_channel * iw * ih * batch_size)).reshape(batch_size, input_channel, ih, iw)
    print("x=\n", x)
    col_x = img2col(x, fh, fw, stride, padding)
    print("col_x=\n", col_x)
    print("w=\n", wb.W)
    col_w = wb.W.reshape(output_channel, -1).T
    print("col_w=\n", col_w)

    # backward
    delta_in = np.array(range(batch_size*output_channel*output_height*output_width)).reshape(batch_size, output_channel, output_height, output_width)
    print("delta_in=\n", delta_in)

    delta_in_2d = np.transpose(delta_in, axes=(0,2,3,1)).reshape(-1, output_channel)
    print("delta_in_2d=\n", delta_in_2d)

    dB = np.sum(delta_in_2d, axis=0, keepdims=True).T / batch_size
    print("dB=\n", dB)
    dW = np.dot(col_x.T, delta_in_2d) / batch_size
    print("dW=\n", dW)
    dW = np.transpose(dW, axes=(1, 0)).reshape(output_channel, input_channel, fh, fw)
    print("dW=\n", dW)
    dcol = np.dot(delta_in_2d, col_w.T)
    print("dcol=\n", dcol)
    delta_out = col2img(dcol, x.shape, fh, fw, stride, padding, output_height, output_width)
    print("delta_out=\n", delta_out)

if __name__ == '__main__':
    understand_4d_col2img_simple()
    understand_4d_col2img_complex()
