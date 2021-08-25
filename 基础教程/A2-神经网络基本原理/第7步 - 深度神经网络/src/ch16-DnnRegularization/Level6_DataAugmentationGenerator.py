# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import cv2
import gzip
import math
import random
import os
import matplotlib.pyplot as plt

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.

from ExtendedDataReader.MnistImageDataReader import *

def LoadData():
    mdr = MnistImageDataReader("image")
    mdr.ReadLessData(1000)
    mdr.NormalizeX()
    return mdr

def mnist_image_reader(path, index):
    """get image data from mnist file

    get the image data from mnist file as described in mnist

    Args:
        path: the path pointing to the data file
        index: a list recording the index of chosen data
    
    Return:
        a numpy array which contain all the image data [length(index), height, width]
    """
    with gzip.open(path) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * 60000 * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data / PIXEL_DEPTH
        data = data.reshape(60000, IMAGE_SIZE, IMAGE_SIZE)
        data = np.array([data[i, :, :] for i in index])
    return data
 
def mnist_label_reader(path, index):
    """get label data from mnist file

    get the label data from mnist file as described in mnist

    Args:
        path: string the path pointing to the data file
        index: a list recording the index of chosen data
    
    Return:
        a numpy array which contain all the label data [size, k]
    """

    with gzip.open(path) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * 60000)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        labels = np.array([labels[i] for i in index])
    return labels
    
def rotate(image, angle):
    """rotate the given image for a given angle

    Args:
        image: numpy array [height, width]
        angle: float from (-180, 180)
    
    Return:
        a numpy array with the same size of image
    """
    height, width = image.shape
    center = (height // 2, width // 2)
    rotation = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(image, rotation, (width, height))
    return rotated_image
# end rotate

def stretch(image, ratio, direction=0):
    """stretch the given image

    stretch the given image according to ratio and direction

    Args:
        image: numpy array [height, width]
        ratio: float from 0 to 2, note how large the image should be stretched
        direction: 0 for vertical direction, 1 for horizontal direction
    
    Return:
        a numpy array with the same size of input image
    """
    height, width = image.shape
    
    if direction == 0:
        new_height = int(height * ratio)
        new_width = width
    else:
        new_height = height
        new_width = int(width * ratio)
    # end if

    resized_image = np.clip(cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC), 0, 1)
    if ratio > 1:
        center_height = new_height // 2
        center_width = new_width // 2
        top_height = center_height - height // 2
        left_width = center_width - width // 2
        return resized_image[top_height:(top_height + height), left_width:(left_width + width)]
    else:
        top_border = (height - new_height) / 2
        left_border = (width - new_width) / 2
        bottom_border = math.ceil(top_border)
        top_border = math.floor(top_border)
        right_border = math.ceil(left_border)
        left_border = math.floor(left_border)
        return cv2.copyMakeBorder(resized_image, top_border, bottom_border, left_border, right_border, cv2.BORDER_CONSTANT, value=0)
    # end if
#end stretch

def translate(image, distance, direction=0):
    """move the given image

    move the given image for several pixels at the given direction

    Args:
        image: numpy array [height, width]
        distance: int, represents how many pixels to move
        direction: 0 for vertical direction, 1 for horizontal direction

    Return:
        a numpy array with the same size of input image
    """
    height, width = image.shape

    if direction == 0:
        M = np.float32([[1, 0, 0], [0, 1, distance]])
    else:
        M = np.float32([[1, 0, distance], [0, 1, 0]])
    # end if

    return cv2.warpAffine(image, M, (width, height))
# end translate

def noise(image, var=0.1):
    """add gaussian noise on the given image

    Args:
        image: numpy array [height, width]
        var: denotes how much noise added on the image
    
    Return:
        a numpy array with the same size of input image
    """
    gaussian_noise = np.random.normal(0, var ** 0.5, image.shape)
    noise_image = image + gaussian_noise
    return np.clip(noise_image, 0, 1)

def generate(subfolder):
    dataReader = LoadData()
    image_list = dataReader.XTrain.reshape(1000,28,28)
    label = dataReader.YTrainRaw

    rotated_image_10 = map(lambda x: rotate(x, 10), image_list)
    rotated_image_minus_10 = map(lambda x: rotate(x, -10), image_list)

    stretch_image_11_v = map(lambda x: stretch(x, 1.2, 0), image_list)
    stretch_image_09_v = map(lambda x: stretch(x, 0.8, 0), image_list)
    stretch_image_11_h = map(lambda x: stretch(x, 1.2, 1), image_list)
    stretch_image_09_h = map(lambda x: stretch(x, 0.8, 1), image_list)

    translate_image_2_v = map(lambda x: translate(x, 2, 0), image_list)
    translate_image_2_h = map(lambda x: translate(x, 2, 1), image_list)

    noise_image = map(lambda x: noise(x), image_list)
    
    all_image = np.concatenate([image_list, list(rotated_image_10), list(rotated_image_minus_10), list(stretch_image_11_v),
                list(stretch_image_09_v), list(stretch_image_11_h), list(stretch_image_09_h),
                list(translate_image_2_v), list(translate_image_2_h), list(noise_image)]) * 255.0
    all_label = np.concatenate([label for i in range(10)])
    all_image = all_image.astype(np.uint8)
    all_label = all_label.astype(np.uint8)

    isExists = os.path.exists(subfolder)
    if not isExists:
        os.makedirs(subfolder)
    np.savez(subfolder + "/data.npz", data=all_image, label=all_label)

if __name__=="__main__":

    generate("augmentation")

    data = np.load("augmentation/data.npz")
    image = data["data"] 
    label = data["label"]
    print(image.shape, label.shape)

    for i in range(3):
        # rotate
        fig = plt.figure(figsize=(10,3))
        axes = plt.subplot(1,3,2)
        axes.imshow(image[i])
        axes = plt.subplot(1,3,1)
        axes.imshow(image[i+1000])
        axes = plt.subplot(1,3,3)
        axes.imshow(image[i+2000])
        plt.suptitle(label[i])
        plt.show()
        # stretch
        fig = plt.figure(figsize=(10,10))
        axes = plt.subplot(3,3,5)
        axes.imshow(image[i])
        axes = plt.subplot(3,3,4)
        axes.imshow(image[i+3000])
        axes = plt.subplot(3,3,6)
        axes.imshow(image[i+4000])
        axes = plt.subplot(3,3,2)
        axes.imshow(image[i+5000])
        axes = plt.subplot(3,3,8)
        axes.imshow(image[i+6000])
        plt.suptitle(label[i])
        plt.show()
        
        # translate
        fig = plt.figure(figsize=(8,8))
        axes = plt.subplot(2,2,1)
        axes.imshow(image[i])
        axes = plt.subplot(2,2,2)
        axes.imshow(image[i+7000])
        axes = plt.subplot(2,2,3)
        axes.imshow(image[i+8000])
        axes = plt.subplot(2,2,4)
        axes.imshow(image[i+9000])
        plt.suptitle(label[i])
        plt.show()
        




