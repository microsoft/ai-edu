# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

#coding=utf-8

import numpy as np
import numba as nb

# 简单地加了个 jit 后的卷积，用数组运算
@nb.jit(nopython=True)
def jit_conv_kernel(x, w, rs, n, n_channels, height, width, num_output_channel, filter_height, filter_width, out_h, out_w):
    for i in range(n):
        for j in range(out_h):
            for p in range(out_w):
                window = x[i, ..., j:j+filter_height, p:p+filter_width]
                for q in range(num_output_channel):
                    rs[i, q, j, p] += np.sum(w[q] * window)
    return rs


@nb.jit(nopython=True)
def jit_conv_kernel2(x, w, rs, batch_size, num_input_channel, input_height, input_width, num_output_channel, filter_height, filter_width, out_h, out_w):
    for i in range(batch_size):
        for j in range(out_h):
            for p in range(out_w):
                for q in range(num_output_channel):
                    for r in range(num_input_channel):
                        for s in range(filter_height):
                            for t in range(filter_width):
                                rs[i, q, j, p] += x[i, r, j+s, p+t] * w[q, r, s, t]
    return rs

@nb.jit(nopython=True)
def conv2d(input_array, kernal, bias, output_array):
    assert(input_array.ndim == 2)
    assert(output_array.ndim == 2)
    assert(kernal.ndim == 2)

    output_height = output_array.shape[0]
    output_width = output_array.shape[1]
    kernal_height = kernal.shape[0]
    kernal_width = kernal.shape[1]

    for i in range(output_height):
        i_start = i
        i_end = i_start + kernal_height
        for j in range(output_width):
            j_start = j
            j_end = j_start + kernal_width
            target_array = input_array[i_start:i_end, j_start:j_end]
            output_array[i,j] = np.sum(target_array * kernal) + bias

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
                            # end fw
                        # end fh
                    # end j
                # end i
            # end ic
        # end oc
    #end bs
    return rs

@nb.jit(nopython=True)
def calculate_output_size(input_h, input_w, filter_h, filter_w, pad=0, stride=1):
    output_h = (input_h - filter_h + 2 * pad) // stride + 1    
    output_w = (input_w - filter_w + 2 * pad) // stride + 1
    return output_h, output_w

# 其实就是calculate_output_size的逆向计算
@nb.jit(nopython=True)
def calculate_padding_size(input_h, input_w, filter_h, filter_w, output_h, output_w, stride):
    pad_h = ((output_h - 1) * stride - input_h + filter_h) / 2
    pad_w = ((output_w - 1) * stride - input_w + filter_w) / 2
    return pad_h, pad_w

"""
对于stride不是1的情况，delta_in数组会比stride是1的情况要小nxn，梯度值不利于对应到输入矩阵上，所以要先转换成stride=1的情况
batch_size: 批大小
input_c: 输入图片的通道数
input_h: 输入图片的高度
input_w: 输入图片的宽度
output_h: 卷积后输出图片的高度(使用设定的stride和padding)
output_w: 卷积后输出图片的宽度(使用设定的stride和padding)
filter_h: 过滤器的高度
filter_w: 过滤器的宽度
padding: 设定的填充值
stride: 设定的步长
"""
#@nb.jit(nopython=True)
def expand_delta_map(dZ, batch_size, input_c, input_h, input_w, output_h, output_w, filter_h, filter_w, padding, stride):
    assert(dZ.ndim == 4)

    if stride == 1:
        dZ_stride_1 = dZ
    else:
        # 假设如果stride等于1时，卷积后输出的图片大小应该是多少，然后根据这个尺寸调整delta_z的大小
        expand_h, expand_w = calculate_output_size(input_h, input_w, filter_h, filter_w, pad, 1)
        # 初始化一个0数组，四维
        dZ_stride_1 = np.zeros((batch_size, input_c, expand_h, expand_w))
        # 把误差值填到当stride=1时的对应位置上
        for bs in range(batch_size):
            for ic in range(input_c):
                for i in range(output_h):
                    for j in range(output_w):
                        ii = i * stride
                        jj = j * stride
                        dZ_stride_1[bs, ic, ii, jj] = dZ[bs, ic, i, j]
                    #end j
                # end i
            # end ic
        # end bs
    # end else

    """
    求本层的输出误差矩阵时，应该用本层的输入误差矩阵互相关计算本层的卷积核的旋转
    由于输出误差矩阵的尺寸必须与本层的输入数据的尺寸一致，所以必须根据卷积核的尺寸，调整本层的输入误差矩阵的尺寸
    """
    pad_h, pad_w = calculate_padding_size(input_h, input_w, filter_h, filter_w, expand_h, expand_w, stride)
    dZ_padded = np.pad(dZ_stride_1, ((0,0),(0,0),(pad_h, pad_h),(pad_w, pad_w)))

    return dZ_padded, dZ_stride_1




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
