
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
import struct


train_folders = {
    "./extended_png_files/Train/add":10,
    "./extended_png_files/Train/minus":11,
    "./extended_png_files/Train/mul":12,
    "./extended_png_files/Train/div":13,
    "./extended_png_files/Train/lp":14,
    "./extended_png_files/Train/rp":15,
    }

test_folders = {
    "./extended_png_files/Test/add":10,
    "./extended_png_files/Test/minus":11,
    "./extended_png_files/Test/mul":12,
    "./extended_png_files/Test/div":13,
    "./extended_png_files/Test/lp":14,
    "./extended_png_files/Test/rp":15,
    }


def get_file_count():
    train_count = 0
    for key in train_folders:
        list = os.listdir(key)
        train_count += len(list)

    test_count = 0
    for key in test_folders:
        list = os.listdir(key)
        test_count += len(list)

    return train_count, test_count

def ReadOneImage(img_file_name):
    img = Image.open(img_file_name)
    a = np.array(img)
    b = 255 - a
    return b.reshape(1,-1)
    
def write_file(folders, image_file, label_file, file_count):
    fp_data = open(image_file, "wb")
    header=struct.pack('>iiii',2053, file_count, 28, 28)
    fp_data.write(header)

    fp_label = open(label_file, "wb")
    header = struct.pack('>ii', 2055, file_count)
    fp_label.write(header)

    count = 0
    for key in folders:
        print(key)
        list = os.listdir(key)
        count += len(list)
        for file in list:
            filepath = os.path.join(key, file)
            array = ReadOneImage(filepath)
            for i in range(784):
                data = struct.pack('B', array[0,i])
                fp_data.write(data)
            # end for
            label = struct.pack('B',folders[key])
            fp_label.write(label)
    fp_data.close()
    fp_label.close()
    print(count, count*784)

if __name__ == "__main__":
    train_count, test_count = get_file_count()
    print(train_count, test_count)
    write_file(train_folders, "train_image_6", "train_label_6", train_count)
    write_file(test_folders, "test_image_6", "test_label_6", test_count)
