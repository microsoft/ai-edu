# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from MiniFramework.EnumDef_6_0 import *
import numba as nb

#@nb.jit(nopython=True)
def img2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    img = input_data
    if pad > 0:
        img = np.zeros((input_data.shape[0],input_data.shape[1],input_data.shape[2]+2*pad,input_data.shape[2]+2*pad))
        img[:,:,pad:pad+input_data.shape[2],pad:pad+input_data.shape[2]] = input_data[:,:]
        #img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
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

#@nb.jit(nopython=True)
def col2img(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
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

if __name__ == '__main__':
    img = np.array(range(9)).reshape(1,1,3,3)
    print(img)
    col = img2col(img, 2, 2, 1, 0)
    print(col)

    img = np.array(range(36)).reshape(2,2,3,3)
    print(img)
    col = img2col(img, 2, 2, 1, 0)
    print(col)
