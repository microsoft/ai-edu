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

def test_performance():
    batch_size = 64
    params = HyperParameters_4_2(
        0.1, 1, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier)
    stride = 1
    padding = 1
    fh = 3
    fw = 3
    input_channel = 3
    output_channel = 4
    iw = 28
    ih = 28
    # 64 个 3 x 28 x 28 的图像输入（模拟 mnist）
    x = np.random.randn(batch_size, input_channel, iw, ih)
    
    c1 = ConvLayer((input_channel,iw,ih), (output_channel,fh,fw), (stride, padding), params)
    c1.initialize("test", "test", False)

    # dry run
    for i in range(5):
        f1 = c1.forward_numba(x)
        delta_in = np.ones((f1.shape))
        b1, dw1, db1 = c1.backward_numba(delta_in, 1)
    # run
    s1 = time.time()
    for i in range(100):
        f1 = c1.forward_numba(x)
        b1, dw1, db1 = c1.backward_numba(delta_in, 1)
    e1 = time.time()
    print("method numba:", e1-s1)

    # dry run
    for i in range(5):
        f2 = c1.forward_img2col(x)
        b2, dw2, db2 = c1.backward_col2img(delta_in, 1)
    # run
    s2 = time.time()
    for i in range(100):
        f2 = c1.forward_img2col(x)
        b2, dw2, db2 = c1.backward_col2img(delta_in, 1)
    e2 = time.time()
    print("method img2col:", e2-s2)

    print("compare correctness of method 1 and method 2:")
    print("forward:", np.allclose(f1, f2, atol=1e-7))
    print("backward:", np.allclose(b1, b2, atol=1e-7))
    print("dW:", np.allclose(dw1, dw2, atol=1e-7))
    print("dB:", np.allclose(db1, db2, atol=1e-7))

if __name__ == '__main__':
    test_performance()
