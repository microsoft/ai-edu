# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import numba as nb
import timeit

# 普通的卷积，用数组运算
def conv_kernel(x, w, rs, n, num_input_channel, height, width, num_output_channel, filter_height, filter_width, out_h, out_w):
    for i in range(n):
        for j in range(out_h):
            for p in range(out_w):
                window = x[i, ..., j:j+filter_height, p:p+filter_width]
                for q in range(num_output_channel):
                    rs[i, q, j, p] += np.sum(w[q] * window)
    return rs

# 逐点运算的卷积，最慢
def conv_kernel2(x, w, rs, n, num_input_channel, height, width, num_output_channel, filter_height, filter_width, out_h, out_w):
    for i in range(n):
        for j in range(out_h):
            for p in range(out_w):
                for q in range(num_output_channel):
                    for r in range(num_input_channel):
                        for s in range(filter_height):
                            for t in range(filter_width):
                                rs[i, q, j, p] += x[i, r, j+s, p+t] * w[q, r, s, t]
    return rs

# 简单地加了个 jit 后的卷积，用数组运算
@nb.jit(nopython=True)
def jit_conv_kernel(x, weights, rs, batch_size, num_input_channel, height, width, num_output_channel, filter_height, filter_width, out_h, out_w):
    for b in range(batch_size):
        for i in range(out_h):
            for j in range(out_w):
                window = x[b, ..., i:i+filter_height, j:j+filter_width]
                for d in range(num_output_channel):
                    rs[b, d, i, j] += np.sum(weights[d] * window)
    return rs

# 加jit, 用逐点运算，会比用数组运算快
@nb.jit(nopython=True)
def jit_conv_kernel2(x, w, rs, n, num_input_channel, height, width, num_output_channel, filter_height, filter_width, out_h, out_w):
    for i in range(n):
        for j in range(out_h):
            for p in range(out_w):
                for q in range(num_output_channel):
                    for r in range(num_input_channel):
                        for s in range(filter_height):
                            for t in range(filter_width):
                                rs[i, q, j, p] += x[i, r, j+s, p+t] * w[q, r, s, t]
    return rs

@nb.jit(nopython=True)
def my_conv_4d(x, weights, rs, batch_size, num_input_channel, height, width, num_output_channel, filter_height, filter_width, out_h, out_w, stride):
    # 输入图片的批大小，通道数，高，宽
    assert(x.ndim == 4)
    # 输入图片的通道数
    assert(x.shape[1] == weights.shape[1])  
    batch_size = x.shape[0]
    num_input_channel = x.shape[1]
    num_output_channel = weights.shape[0]
    filter_height = weights.shape[2]
    filter_width = weights.shape[3]

    for bs in range(batch_size):
        for oc in range(num_output_channel):
            for ic in range(num_input_channel):
                for i in range(out_h):
                    for j in range(out_w):
                        ii = i * stride
                        jj = j * stride
                        for fh in range(filter_height):
                            for fw in range(filter_width):
                                rs[bs,oc,i,j] += x[bs,ic,ii+fh,jj+fw] * weights[oc,ic,fh,fw]
        # end oc
    #end b
    return rs

def conv1():
    rs = np.zeros([n, num_output_channel, out_h, out_w])
    conv_kernel2(x, w, rs, *args)
    return rs

def conv2():
    rs = np.zeros([n, num_output_channel, out_h, out_w])
    conv_kernel(x, w, rs, *args)
    return rs


def conv3():
    rs = np.zeros([n, num_output_channel, out_h, out_w])
    #jit_conv_kernel(x, w, rs, *args)
    my_conv_4d(x, w, rs, *args)
    return rs

def conv4():
    rs = np.zeros([n, num_output_channel, out_h, out_w])
    jit_conv_kernel2(x, w, rs, *args)
    return rs

#@nb.jit(nopython=True)
def cal5(w, num_output_channel, col_x, out_h, out_w, N):
    col_x = col_x.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    col_w = w.reshape(num_output_channel, -1).T
    out = np.dot(col_x, col_w)
    rs = out.reshape(n, out_h, out_w, -1).transpose(0,3,1,2)
    return rs

#@nb.jit(nopython=True)
def conv5():
    stride=3
    pad=0

    # im2col
    N, C, H, W = x.shape
    out_h = (H + 2*pad - filter_height)//stride + 1
    out_w = (W + 2*pad - filter_width)//stride + 1
    #img = np.pad(x, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    img = x
    col_x = np.zeros((N, C, filter_height, filter_width, out_h, out_w))

    for i in range(filter_height):
        i_max = i + stride*out_h
        for j in range(filter_width):
            j_max = j + stride*out_w
            col_x[:, :, i, j, :, :] = img[:, :, i:i_max:stride, j:j_max:stride]

    #col_x = col_x.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)

#    col_x = im2col(x, filter_height, filter_width, 1, 0)
    rs = cal5(w, num_output_channel, col_x, out_h, out_w, N)

    #col_w = w.reshape(num_output_channel, -1).T
    #out = np.dot(col_x, col_w)
    #rs = out.reshape(n, out_h, out_w, -1).transpose(0,3,1,2)

    return rs




def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

if __name__ == '__main__':
       
    # 64 个 3 x 28 x 28 的图像输入（模拟 mnist）
    x = np.random.randn(64, 3, 28, 28)
    # 16 个 5 x 5 的 kernel
    w = np.random.randn(16, x.shape[1], 5, 5)
    stride = 3
    pad = 0
    n, num_input_channel, height, width = x.shape
    num_output_channel, _, filter_height, filter_width = w.shape
    out_h = (height - filter_height + 2*pad)//stride + 1
    out_w = (width - filter_width + 2*pad)//stride + 1
    args = (n, num_input_channel, height, width, num_output_channel, filter_height, filter_width, out_h, out_w, stride)
    

   #  print(np.linalg.norm(conv1() - conv2()).ravel())

    #t1 = timeit.repeat('conv1()','from __main__ import conv1', number=3)
    #t2 = timeit.repeat('conv2()','from __main__ import conv2', number=3)
   
    
    rs4 = conv5()
    rs3 = conv3()
    result = np.allclose(rs3, rs4, rtol=1e-05, atol=1e-05)
    print(result)
    print(np.linalg.norm(rs3 - rs4).ravel())
    print((rs4-rs4).sum())
    
    num = 5

#    t1 = timeit.timeit('conv1()','from __main__ import conv1', number=1)
#    print("t1:", t1, t1/1)
    
 #   t2 = timeit.timeit('conv2()','from __main__ import conv2', number=num)
#    print("t2:", t2, t2/n)
    """
    t3 = timeit.timeit('conv3()','from __main__ import conv3', number=num)
    print("t3:", t3, t3/num)

    t4 = timeit.timeit('conv4()','from __main__ import conv4', number=num)
    print("t4:", t4, t4/num)

    t5 = timeit.timeit('conv5()','from __main__ import conv5', number=num)
    print("t5:", t5, t5/num)
    """

#    print(t1/t2)
#    print(t2/t3)
#    print(t3/t4)
#    print(t4/t5)


'''
result:
逐点运算
t1: 77.1346427 15.42692854
数组运算
t2: 13.6764096 2.73528192
数组运算 with jit
t3: 1.4621606000000043 0.29243212000000085
逐点运算 with jit
t4: 0.4403588000000127 0.08807176000000254
t1/t2: 5.63997752012341
t2/t3: 9.353561845395069
t3/t4: 3.3203846499717096    
'''
