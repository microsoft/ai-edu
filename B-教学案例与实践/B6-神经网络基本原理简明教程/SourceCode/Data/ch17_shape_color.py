# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from PIL import Image, ImageDraw, ImageColor
import numpy as np
import matplotlib.pyplot as plt


train_data_name = "../../data/ch17.train_shape_color.npz"
test_data_name = "../../data/ch17.test_shape_color.npz"


def circle(drawObj,color):
    x0 = np.random.randint(0,14)
    y0 = np.random.randint(0,14)
    x1 = np.random.randint(14,28)
    y1 = np.random.randint(14,28)
    drawObj.ellipse([x0,y0,x1,y1], fill=color, outline=color)
    
def rectangle(drawObj,color):
    x0 = np.random.randint(0,14)
    y0 = np.random.randint(0,14)
    x1 = np.random.randint(14,28)
    y1 = np.random.randint(14,28)
    drawObj.rectangle([x0,y0,x1,y1], fill=color, outline=color)

def triangle(drawObj,color):
    x0 = np.random.randint(0,14)
    y0 = np.random.randint(0,14)
    x1 = np.random.randint(14,28)
    y1 = np.random.randint(0,14)
    x2 = np.random.randint(0,14)
    y2 = np.random.randint(14,28)
    x3 = np.random.randint(14,28)
    y3 = np.random.randint(14,28)
    r = np.random.randint(0,4)
    if r == 0:
        drawObj.polygon([x0,y0,x1,y1,x2,y2], fill=color, outline=color)
    elif r == 1:
        drawObj.polygon([x0,y0,x1,y1,x3,y3], fill=color, outline=color)
    elif r == 2:
        drawObj.polygon([x0,y0,x2,y2,x3,y3], fill=color, outline=color)
    else:
        drawObj.polygon([x1,y1,x2,y2,x3,y3], fill=color, outline=color)

def shape_color(count):
    colors = [(255,0,0), (0,255,0), (0,0,255)]
    shapes = [circle, rectangle, triangle]

    images = np.empty((count*9,3,28,28))
    labels = np.empty((count*9,1))

    for i in range(3):
        color = colors[i]
        for j in range(3):
            shape = shapes[j]
            label = i * 3 + j
            for k in range(count):
                img = Image.new("RGB", [28,28], "black")
                drawObj = ImageDraw.Draw(img)
                shape(drawObj, color)
                id = i * 3 * count + j * count + k
                images[id] = np.array(img).transpose(2,0,1)
                labels[id] = label
            #end for
        #endfor
    #endfor
    return images, labels


if __name__ == '__main__':
    train_x, train_y = shape_color(600)
    test_x, test_y = shape_color(100)

    seed = np.random.randint(0,100)
    np.random.seed(seed)
    np.random.shuffle(train_x)
    np.random.seed(seed)
    np.random.shuffle(train_y)
    np.random.seed(seed)
    np.random.shuffle(test_x)
    np.random.seed(seed)
    np.random.shuffle(test_y)

    np.savez(train_data_name, data=train_x, label=train_y)
    np.savez(test_data_name, data=test_x, label=test_y)
