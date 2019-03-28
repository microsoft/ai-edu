# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.


# output array: 768 x num_images
def ReadImageFile(image_file_name):
    f = open(image_file_name, "rb")
    a = f.read(4)
    b = f.read(4)
    num_images = int.from_bytes(b, byteorder='big')
    c = f.read(4)
    num_rows = int.from_bytes(c, byteorder='big')
    d = f.read(4)
    num_cols = int.from_bytes(d, byteorder='big')

    image_size = num_rows * num_cols    # 784
    fmt = '>' + str(image_size) + 'B'
    image_data = np.empty((image_size, num_images)) # 784 x M
    for i in range(num_images):
        bin_data = f.read(image_size)
        unpacked_data = struct.unpack(fmt, bin_data)
        array_data = np.array(unpacked_data)
        array_data2 = array_data.reshape((image_size, 1))
        image_data[:,i] = array_data
    f.close()
    return image_data

def ReadLabelFile(lable_file_name, num_output):
    f = open(lable_file_name, "rb")
    f.read(4)
    a = f.read(4)
    num_labels = int.from_bytes(a, byteorder='big')

    fmt = '>B'
    label_data = np.zeros((num_output, num_labels))   # 10 x M
    for i in range(num_labels):
        bin_data = f.read(1)
        unpacked_data = struct.unpack(fmt, bin_data)[0]
        label_data[unpacked_data,i] = 1
    f.close()
    return label_data

def NormalizeByRow(X):
    X_NEW = np.zeros(X.shape)
    # get number of features
    n = X.shape[0]
    for i in range(n):
        x_row = X[i,:]
        x_max = np.max(x_row)
        x_min = np.min(x_row)
        if x_max != x_min:
            x_new = (x_row - x_min)/(x_max-x_min)
            X_NEW[i,:] = x_new
    return X_NEW

def NormalizeData(X):
    X_NEW = np.zeros(X.shape)
    x_max = np.max(X)
    x_min = np.min(X)
    X_NEW = (X - x_min)/(x_max-x_min)
    return X_NEW
