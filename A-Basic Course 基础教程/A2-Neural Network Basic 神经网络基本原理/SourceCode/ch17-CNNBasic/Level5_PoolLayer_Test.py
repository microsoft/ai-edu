# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy
import numba
import time

from MiniFramework.PoolingLayer import *


def understand_img2col():
    batch_size = 3
    input_channel = 2
    input_width = 4
    input_height = 4
    pool_height = 2
    pool_width = 2
    pool_size = pool_height * pool_width
    stride = 2
    output_height = (input_height - pool_height) // stride + 1
    output_width = (input_width - pool_width) // stride + 1
    padding = 0

    x = np.array(range(batch_size*input_channel*input_height*input_width)).reshape(batch_size, input_channel, input_height, input_width)
    print(x)

    # forward
    N, C, H, W = x.shape
    col = img2col(x, pool_height, pool_width, stride, padding)
    print("col=", col)
    col_x = col.reshape(-1, pool_height * pool_width)
    print("col_x=", col_x)
    arg_max = np.argmax(col_x, axis=1)
    print("arg_max=", arg_max)
    out1 = np.max(col_x, axis=1)
    print("out1=", out1)
    out2 = out1.reshape(N, output_height, output_width, C)
    print("out2=", out2)
    z = np.transpose(out2, axes=(0,3,1,2))
    print(z)

    # backward
    delta_in = np.ones(z.shape)
    dout = np.transpose(delta_in, (0,2,3,1))
    dmax = np.zeros((dout.size, pool_size))
    dmax[np.arange(arg_max.size), arg_max.flatten()] = dout.flatten()
    dmax = dmax.reshape(dout.shape + (pool_size,))
    dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
    dx = col2img(dcol, x.shape, pool_height, pool_width, stride, padding)
    print(dx)

def test_performance():
    batch_size = 64
    input_channel = 3
    iw = 28
    ih = 28
    x = np.random.randn(batch_size, input_channel, iw, ih)

    p = PoolingLayer((input_channel,iw,ih),(2,2),2, "MAX")
    # dry run
    f1 = p.forward_numba(x, True)
    delta_in = np.random.random(f1.shape)
    # run
    s1 = time.time()
    for i in range(5000):
        f1 = p.forward_numba(x, True)
        b1 = p.backward_numba(delta_in, 0)
    e1 = time.time()
    print("Elapsed of numba:", e1-s1)

    # dry run
    f2 = p.forward_img2col(x, True)
    b2 = p.backward_col2img(delta_in, 1)
    # run
    s2 = time.time()
    for i in range(5000):
        f2 = p.forward_img2col(x, True)
        b2 = p.backward_col2img(delta_in, 1)
    e2 = time.time()
    print("Elapsed of img2col:", e2-s2)

    print("forward:", np.allclose(f1, f2, atol=1e-7))
    print("backward:", np.allclose(b1, b2, atol=1e-7))

if __name__ == '__main__':
    test_performance()
    #understand_img2col()

