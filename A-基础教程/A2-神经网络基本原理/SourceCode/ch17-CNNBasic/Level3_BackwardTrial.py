# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from MiniFramework.jit_utility import *
from matplotlib import pyplot as plt
import cv2

circle_pic = "circle.png"

def normalize(x):
    min = np.min(x)
    max = np.max(x)
    x_n = (x - min)/(max - min)
    return x_n

def create_zero_array(x,w):
    out_h, out_w = calculate_output_size(x.shape[0], x.shape[1], w.shape[0], w.shape[1], 0, 1)
    output = np.zeros((out_h, out_w))
    return output

def train(x, w, b, y):
    output = create_zero_array(x, w)
    for i in range(10000):
        # forward
        jit_conv_2d(x, w, b, output)
        # loss
        t1 = (output - y)
        m = t1.shape[0]*t1.shape[1]
        LOSS = np.multiply(t1, t1)
        loss = np.sum(LOSS)/2/m
        print(i,loss)
        if loss < 1e-7:
            break
        # delta
        delta = output - y
        # backward
        dw = np.zeros(w.shape)
        jit_conv_2d(x, delta, b, dw)
        w = w - 0.5 * dw/m
    #end for
    return w


def create_sample_image():
    img_color = cv2.imread(circle_pic)
    img_gray = normalize(cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY))
    w = np.array([[0,-1,0],
                  [0, 2,0],
                  [0,-1,0]])
    b = 0
    y = create_zero_array(img_gray, w)
    jit_conv_2d(img_gray, w, b, y)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6,4))
    ax[0].imshow(img_gray, cmap='gray')
    ax[0].set_title("source")
    ax[1].imshow(y, cmap='gray')
    ax[1].set_title("target")
    plt.show()
    return img_gray, w, y

def show_result(img_gray, w_true, w_result):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6,4))
    y = create_zero_array(img_gray, w_true)
    jit_conv_2d(img_gray, w_true, 0, y)
    ax[0].imshow(y, cmap='gray')
    ax[0].set_title("true")
    z = create_zero_array(img_gray, w_result)
    jit_conv_2d(img_gray, w_result, 0, z)
    ax[1].imshow(z, cmap='gray')
    ax[1].set_title("result")
    plt.show()

if __name__ == '__main__':
    # 创建样本数据
    x, w_true, y = create_sample_image()
    # 随机初始化卷积核
    w_init = np.random.normal(0, 0.1, w_true.shape)
    # 训练
    w_result = train(x,w_init,0,y)
    # 打印比较真实卷积核值和训练出来的卷积核值
    print("w_true:\n", w_true)
    print("w_result:\n", w_result)
    # 用训练出来的卷积核值对原始图片进行卷积
    y_hat = np.zeros(y.shape)
    jit_conv_2d(x, w_true, 0, y_hat)
    # 与真实的卷积核的卷积结果比较
    show_result(x, w_true, w_result)
    # 比较卷积核值的差异核卷积结果的差异
    print("w allclose:", np.allclose(w_true, w_result, atol=1e-2))
    print("y allclose:", np.allclose(y, y_hat, atol=1e-2))
