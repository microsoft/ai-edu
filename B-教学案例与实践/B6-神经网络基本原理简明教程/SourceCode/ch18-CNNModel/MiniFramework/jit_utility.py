# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from MiniFramework.EnumDef_6_0 import *
import numba as nb
from numba import float32, int32

@nb.jit(nopython=True)
def jit_maxpool_forward(x, batch_size, input_c, output_h, output_w, pool_h, pool_w, pool_stride):
    z = np.zeros((batch_size, input_c, output_h, output_w))
    for b in range(batch_size):
        for c in range(input_c):
            for i in range(output_h):
                i_start = i * pool_stride
                i_end = i_start + pool_h
                for j in range(output_w):
                    j_start = j * pool_stride
                    j_end = j_start + pool_w
                    target_array = x[b,c,i_start:i_end, j_start:j_end]
                    t = np.max(target_array)
                    z[b,c,i,j] = t

    return z

@nb.jit(nopython=True)
def jit_maxpool_backward(x, delta_in, batch_size, input_c, output_h, output_w, pool_h, pool_w, pool_stride):
    delta_out = np.zeros(x.shape)
    for b in range(batch_size):
        for c in range(input_c):
            for i in range(output_h):
                i_start = i * pool_stride
                i_end = i_start + pool_h
                for j in range(output_w):
                    j_start = j * pool_stride
                    j_end = j_start + pool_w
                    m,n = jit_get_max_index(x[b,c], i_start, i_end, j_start, j_end)
                    delta_out[b,c,m,n] = delta_in[b,c,i,j]

    return delta_out

@nb.jit(nopython=True)
def jit_get_max_index(input, i_start, i_end, j_start, j_end):
    assert(input.ndim == 2)
    max_i = i_start
    max_j = j_start
    max_value = input[i_start,j_start]
    for i in range(i_start,i_end):
        for j in range(j_start,j_end):
            if input[i,j] > max_value:
                max_value = input[i,j]
                max_i, max_j = i, j

    return max_i, max_j

@nb.jit(nopython=True)
def jit_conv_2d(input_array, kernal, bias, output_array):
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
    #end bs
    return rs

# 标准卷积后输出尺寸计算公式
@nb.jit(nopython=True)
def calculate_output_size(input_h, input_w, filter_h, filter_w, padding, stride=1):
    output_h = (input_h - filter_h + 2 * padding) // stride + 1    
    output_w = (input_w - filter_w + 2 * padding) // stride + 1
    return (output_h, output_w)

# 其实就是calculate_output_size的逆向计算
@nb.jit(nopython=True)
def calculate_padding_size(input_h, input_w, filter_h, filter_w, output_h, output_w, stride=1):
    pad_h = ((output_h - 1) * stride - input_h + filter_h) // 2
    pad_w = ((output_w - 1) * stride - input_w + filter_w) // 2
    return (pad_h, pad_w)

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
@nb.jit(nopython=True)
def expand_delta_map(dZ, batch_size, input_c, input_h, input_w, output_h, output_w, filter_h, filter_w, padding, stride):
    assert(dZ.ndim == 4)
    expand_h = 0
    expand_w = 0
    if stride == 1:
        dZ_stride_1 = dZ
        expand_h = dZ.shape[2]
        expand_w = dZ.shape[3]
    else:
        # 假设如果stride等于1时，卷积后输出的图片大小应该是多少，然后根据这个尺寸调整delta_z的大小
        (expand_h, expand_w) = calculate_output_size(input_h, input_w, filter_h, filter_w, padding, 1)
        # 初始化一个0数组，四维
        #dZ_stride_1 = np.zeros((batch_size, input_c, expand_h, expand_w)).astype(np.float32)
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
    return dZ_stride_1

@nb.jit(nopython=True)
def calcalate_weights_grad(x, dz, batch_size, output_c, input_c, filter_h, filter_w, dW, dB):
    for bs in range(batch_size):
        for oc in range(output_c):   # == kernal count
            for ic in range(input_c):    # == filter count
                w_grad = np.zeros((filter_h, filter_w)).astype(np.float32)
                #w_grad = np.zeros((filter_h, filter_w))
                jit_conv_2d(x[bs,ic], dz[bs,oc], 0, w_grad)
                dW[oc,ic] += w_grad
            #end ic
            dB[oc] += dz[bs,oc].sum()
        #end oc
    #end bs
    return (dW, dB)

@nb.jit(nopython=True)
def calculate_delta_out(dz, rot_weights, batch_size, num_input_channel, num_output_channel, input_height, input_width, delta_out):
    for bs in range(batch_size):
        for oc in range(num_output_channel):    # == kernal count
            delta_per_input = np.zeros((input_height, input_width)).astype(np.float32)
            #delta_per_input = np.zeros((input_height, input_width))
            for ic in range(num_input_channel): # == filter count
                jit_conv_2d(dz[bs,oc], rot_weights[oc,ic], 0, delta_per_input)
                delta_out[bs,ic] += delta_per_input
            #END IC
        #end oc
    #end bs
    return delta_out

@nb.jit(nopython=True)
def calculate_output_size(input_h, input_w, filter_h, filter_w, padding, stride=1):
    output_h = (input_h - filter_h + 2 * padding) // stride + 1    
    output_w = (input_w - filter_w + 2 * padding) // stride + 1
    return (output_h, output_w)


#@nb.jit(nopython=True)
def img2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    img = input_data
    if pad > 0:
        #img = np.zeros((input_data.shape[0],input_data.shape[1],input_data.shape[2]+2*pad,input_data.shape[2]+2*pad))
        #img[:,:,pad:pad+input_data.shape[2],pad:pad+input_data.shape[2]] = input_data[:,:]
        img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
        #img = np.pad(input_data,mode="constant",constant_value=0, pad_width=(0,0,0,0,pad,pad,pad,pad))
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for i in range(filter_h):
        i_max = i + stride*out_h
        for j in range(filter_w):
            j_max = j + stride*out_w
            col[:, :, i, j, :, :] = img[:, :, i:i_max:stride, j:j_max:stride]
        #end for
    #end for
    col = np.transpose(col, axes=(0, 4, 5, 1, 2, 3)).reshape(N*out_h*out_w, -1)
    return col


def col2img(col, input_shape, filter_h, filter_w, stride, pad, out_h, out_w):
    N, C, H, W = input_shape
    tmp1 = col.reshape(N, out_h, out_w, C, filter_h, filter_w)
    tmp2 = np.transpose(tmp1, axes=(0, 3, 4, 5, 1, 2))
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for i in range(filter_h):
        i_max = i + stride*out_h
        for j in range(filter_w):
            j_max = j + stride*out_w
            img[:, :, i:i_max:stride, j:j_max:stride] += tmp2[:, :, i, j, :, :]
        #end for
    #end for
    return img[:, :, pad:H + pad, pad:W + pad]


def col2img2(col, input_shape, filter_h, filter_w, stride, pad, out_h, out_w):
    N, C, H, W = input_shape
    tmp1 = col.reshape(N, out_h, out_w, C, filter_h, filter_w)
    tmp2 = np.transpose(tmp1, axes=(0, 3, 4, 5, 1, 2))
    a = fill(filter_h, filter_w, stride, out_h, out_w, pad, tmp2, H, W, N, C)
    return a

@nb.jit(nopython=True)
def fill(filter_h, filter_w, stride, out_h, out_w, pad, tmp2, H, W, N, C):
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for i in range(filter_h):
        i_max = i + stride*out_h
        for j in range(filter_w):
            j_max = j + stride*out_w
            img[:, :, i:i_max:stride, j:j_max:stride] += tmp2[:, :, i, j, :, :]
        #end for
    #end for
    return img[:, :, pad:H + pad, pad:W + pad]



if __name__ == '__main__':

    x = np.array([1,1,1,2,2,2,3,3,3,
                  3,3,3,2,2,2,1,1,1,
                  1,1,1,2,2,2,3,3,3,
                  3,3,3,2,2,2,1,1,1,
                  1,1,1,2,2,2,3,3,3,
                  3,3,3,2,2,2,1,1,1]).reshape(2,3,3,3)
    print(x)
    f = np.array([1,1,1,1,
                  2,2,2,2,
                  3,3,3,3,
                  3,3,3,3,
                  2,2,2,2,
                  1,1,1,1]).reshape(2,3,2,2)
    print(f)
    stride=1
    padding=0
    bias=np.zeros((f.shape[0],1)).astype(np.float32)
    (out_h, out_w) = calculate_output_size(x.shape[2], x.shape[3], f.shape[2], f.shape[3], padding, stride)
    z = jit_conv_4d(x, f, bias, out_h, out_w, stride=1)
    print("------------------")
    print(z)
    print(z.shape)


    a= max_pool_forward(z, 2, 2, 1, 1, 2, 2, 2)
    print(a)