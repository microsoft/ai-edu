# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import numba as nb
import timeit

# 普通的卷积，用数组运算
def conv_kernel(x, w, rs, n, n_channels, height, width, n_filters, filter_height, filter_width, out_h, out_w):
    for i in range(n):
        for j in range(out_h):
            for p in range(out_w):
                window = x[i, ..., j:j+filter_height, p:p+filter_width]
                for q in range(n_filters):
                    rs[i, q, j, p] += np.sum(w[q] * window)
    return rs

# 逐点运算的卷积，最慢
def conv_kernel2(x, w, rs, n, n_channels, height, width, n_filters, filter_height, filter_width, out_h, out_w):
    for i in range(n):
        for j in range(out_h):
            for p in range(out_w):
                for q in range(n_filters):
                    for r in range(n_channels):
                        for s in range(filter_height):
                            for t in range(filter_width):
                                rs[i, q, j, p] += x[i, r, j+s, p+t] * w[q, r, s, t]
    return rs

# 简单地加了个 jit 后的卷积，用数组运算
@nb.jit(nopython=True)
def jit_conv_kernel(x, w, rs, n, n_channels, height, width, n_filters, filter_height, filter_width, out_h, out_w):
    for i in range(n):
        for j in range(out_h):
            for p in range(out_w):
                window = x[i, ..., j:j+filter_height, p:p+filter_width]
                for q in range(n_filters):
                    rs[i, q, j, p] += np.sum(w[q] * window)
    return rs

# 加jit, 用逐点运算，会比用数组运算快
@nb.jit(nopython=True)
def jit_conv_kernel2(x, w, rs, n, n_channels, height, width, n_filters, filter_height, filter_width, out_h, out_w):
    for i in range(n):
        for j in range(out_h):
            for p in range(out_w):
                for q in range(n_filters):
                    for r in range(n_channels):
                        for s in range(filter_height):
                            for t in range(filter_width):
                                rs[i, q, j, p] += x[i, r, j+s, p+t] * w[q, r, s, t]
    return rs

def conv1():
    conv_kernel2(x, w, rs, *args)
    return rs

def conv2():
    conv_kernel(x, w, rs, *args)
    return rs


def conv3():
    jit_conv_kernel(x, w, rs, *args)
    return rs

def conv4():
    jit_conv_kernel2(x, w, rs, *args)
    return rs

#@nb.jit(nopython=True)
def cal5(w, n_filters, col_x, out_h, out_w, N):
    col_x = col_x.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    col_w = w.reshape(n_filters, -1).T
    out = np.dot(col_x, col_w)
    rs = out.reshape(n, out_h, out_w, -1).transpose(0,3,1,2)
    return rs

#@nb.jit(nopython=True)
def conv5():
    stride=1
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
    rs = cal5(w, n_filters, col_x, out_h, out_w, N)

    #col_w = w.reshape(n_filters, -1).T
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

    n, n_channels, height, width = x.shape
    n_filters, _, filter_height, filter_width = w.shape
    out_h = height - filter_height + 1
    out_w = width - filter_width + 1
    rs = np.zeros([n, n_filters, out_h, out_w])
    args = (n, n_channels, height, width, n_filters, filter_height, filter_width, out_h, out_w)
    

   #  print(np.linalg.norm(conv1() - conv2()).ravel())

    #t1 = timeit.repeat('conv1()','from __main__ import conv1', number=3)
    #t2 = timeit.repeat('conv2()','from __main__ import conv2', number=3)
   
    
    rs5 = conv5()
    rs3 = conv3()
    result = np.allclose(rs5, rs3, rtol=1e-05, atol=1e-05)
    print(result)
    print(np.linalg.norm(rs3 - rs5).ravel())
    
    num = 5

#    t1 = timeit.timeit('conv1()','from __main__ import conv1', number=1)
#    print("t1:", t1, t1/1)
    
 #   t2 = timeit.timeit('conv2()','from __main__ import conv2', number=num)
#    print("t2:", t2, t2/n)
    
    t3 = timeit.timeit('conv3()','from __main__ import conv3', number=num)
    print("t3:", t3, t3/num)

    t4 = timeit.timeit('conv4()','from __main__ import conv4', number=num)
    print("t4:", t4, t4/num)

    t5 = timeit.timeit('conv5()','from __main__ import conv5', number=num)
    print("t5:", t5, t5/num)


#    print(t1/t2)
#    print(t2/t3)
    print(t3/t4)
    print(t4/t5)


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
