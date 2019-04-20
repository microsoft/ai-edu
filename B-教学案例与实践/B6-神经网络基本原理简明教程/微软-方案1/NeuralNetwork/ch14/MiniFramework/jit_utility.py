# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

#coding=utf-8

import numpy as np
import numba as nb

# 简单地加了个 jit 后的卷积，用数组运算
@nb.jit(nopython=True)
def jit_conv_kernel(x, w, rs, n, n_channels, height, width, num_output_channels, filter_height, filter_width, out_h, out_w):
    for i in range(n):
        for j in range(out_h):
            for p in range(out_w):
                window = x[i, ..., j:j+filter_height, p:p+filter_width]
                for q in range(num_output_channels):
                    rs[i, q, j, p] += np.sum(w[q] * window)
    return rs


@nb.jit(nopython=True)
def jit_conv_kernel2(x, w, rs, batch_size, num_input_channels, input_height, input_width, num_output_channels, filter_height, filter_width, out_h, out_w):
    for i in range(batch_size):
        for j in range(out_h):
            for p in range(out_w):
                for q in range(num_output_channels):
                    for r in range(num_input_channels):
                        for s in range(filter_height):
                            for t in range(filter_width):
                                rs[i, q, j, p] += x[i, r, j+s, p+t] * w[q, r, s, t]
    return rs

@nb.jit(nopython=True)
def jit_conv_4d(x, weights, bias, out_h, out_w, stride):
    # 输入图片的批大小，通道数，高，宽
    assert(x.ndim == 4)
    # 输入图片的通道数
    assert(x.shape[1] == weights.shape[1])  
    batch_size = x.shape[0]
    num_input_channels = x.shape[1]
    num_output_channels = weights.shape[0]
    filter_height = weights.shape[2]
    filter_width = weights.shape[3]
    rs = np.zeros((batch_size, num_output_channels, out_h, out_w))

    for bs in range(batch_size):
        for oc in range(num_output_channels):
            rs[bs,oc] += bias[oc]
            for ic in range(num_input_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        ii = i * stride
                        jj = j * stride
                        for fh in range(filter_height):
                            for fw in range(filter_width):
                                rs[bs,oc,i,j] += x[bs,ic,ii+fh,jj+fw] * weights[oc,ic,fh,fw]
                            # end fw
                        # end fh
                    # end j
                # end i
            # end ic
        # end oc
    #end bs
    return rs

@nb.jit(nopython=True)
def calculate_output_size(input_h, input_w, filter_h, filter_w, pad, stride):
    output_h = (input_h - filter_h + 2 * pad) // stride + 1    
    output_w = (input_w - filter_w + 2 * pad) // stride + 1
    return output_h, output_w

#@nb.jit(nopython=True)
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    img = input_data
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

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
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for i in range(filter_h):
        i_max = i + stride*out_h
        for j in range(filter_w):
            j_max = j + stride*out_w
            img[:, :, i:i_max:stride, j:j_max:stride] += col[:, :, j, i, :, :]
        #end for
    #end for
    return img[:, :, pad:H + pad, pad:W + pad]