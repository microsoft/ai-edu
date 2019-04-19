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
    x = np.random.randn(64, 3, 28, 28).astype(np.float32)
    # 16 个 5 x 5 的 kernel
    w = np.random.randn(16, x.shape[1], 5, 5).astype(np.float32)

    n, n_channels, height, width = x.shape
    n_filters, _, filter_height, filter_width = w.shape
    out_h = height - filter_height + 1
    out_w = width - filter_width + 1
    args = (n, n_channels, height, width, n_filters, filter_height, filter_width, out_h, out_w)

    n, n_filters = args[0], args[4]
    out_h, out_w = args[-2:]
    rs = np.zeros([n, n_filters, out_h, out_w], dtype=np.float32)
    conv_kernel2(x, w, rs, *args)
    return rs

def conv2():
    x = np.random.randn(64, 3, 28, 28).astype(np.float32)
    # 16 个 5 x 5 的 kernel
    w = np.random.randn(16, x.shape[1], 5, 5).astype(np.float32)

    n, n_channels, height, width = x.shape
    n_filters, _, filter_height, filter_width = w.shape
    out_h = height - filter_height + 1
    out_w = width - filter_width + 1
    args = (n, n_channels, height, width, n_filters, filter_height, filter_width, out_h, out_w)

    n, n_filters = args[0], args[4]
    out_h, out_w = args[-2:]
    rs = np.zeros([n, n_filters, out_h, out_w], dtype=np.float32)
    conv_kernel(x, w, rs, *args)
    return rs


def conv3():
    x = np.random.randn(64, 3, 28, 28).astype(np.float32)
    # 16 个 5 x 5 的 kernel
    w = np.random.randn(16, x.shape[1], 5, 5).astype(np.float32)

    n, n_channels, height, width = x.shape
    n_filters, _, filter_height, filter_width = w.shape
    out_h = height - filter_height + 1
    out_w = width - filter_width + 1
    args = (n, n_channels, height, width, n_filters, filter_height, filter_width, out_h, out_w)

    n, n_filters = args[0], args[4]
    out_h, out_w = args[-2:]
    rs = np.zeros([n, n_filters, out_h, out_w], dtype=np.float32)
    jit_conv_kernel(x, w, rs, *args)
    return rs

def conv4():
    x = np.random.randn(64, 3, 28, 28).astype(np.float32)
    # 16 个 5 x 5 的 kernel
    w = np.random.randn(16, x.shape[1], 5, 5).astype(np.float32)

    n, n_channels, height, width = x.shape
    n_filters, _, filter_height, filter_width = w.shape
    out_h = height - filter_height + 1
    out_w = width - filter_width + 1
    args = (n, n_channels, height, width, n_filters, filter_height, filter_width, out_h, out_w)

    n, n_filters = args[0], args[4]
    out_h, out_w = args[-2:]
    rs = np.zeros([n, n_filters, out_h, out_w], dtype=np.float32)
    jit_conv_kernel2(x, w, rs, *args)
    return rs


if __name__ == '__main__':

        
    # 64 个 3 x 28 x 28 的图像输入（模拟 mnist）
    x = np.random.randn(64, 3, 28, 28).astype(np.float32)
    # 16 个 5 x 5 的 kernel
    w = np.random.randn(16, x.shape[1], 5, 5).astype(np.float32)

    n, n_channels, height, width = x.shape
    n_filters, _, filter_height, filter_width = w.shape
    out_h = height - filter_height + 1
    out_w = width - filter_width + 1
    args = (n, n_channels, height, width, n_filters, filter_height, filter_width, out_h, out_w)
    
   #  print(np.linalg.norm(conv1() - conv2()).ravel())

    #t1 = timeit.repeat('conv1()','from __main__ import conv1', number=3)
    #t2 = timeit.repeat('conv2()','from __main__ import conv2', number=3)
   
    n = 5

    t1 = timeit.timeit('conv1()','from __main__ import conv1', number=2)
    print(t1, t1/n)
    
    t2 = timeit.timeit('conv2()','from __main__ import conv2', number=n)
    print(t2, t2/n)
    
    t3 = timeit.timeit('conv3()','from __main__ import conv3', number=n)
    print(t3, t3/n)

    t4 = timeit.timeit('conv4()','from __main__ import conv4', number=n)
    print(t4, t4/n)

    print(t1/t2)
    print(t2/t3)
    print(t3/t4)


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
