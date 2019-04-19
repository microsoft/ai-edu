# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

#coding=utf-8

import numpy as np
import numba as nb

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


@nb.jit(nopython=True)
def jit_conv_kernel2(x, w, rs, batch_size, num_input_channels, input_height, input_width, n_filters, filter_height, filter_width, out_h, out_w):
    for i in range(batch_size):
        for j in range(out_h):
            for p in range(out_w):
                for q in range(n_filters):
                    for r in range(num_input_channels):
                        for s in range(filter_height):
                            for t in range(filter_width):
                                rs[i, q, j, p] += x[i, r, j+s, p+t] * w[q, r, s, t]
    return rs