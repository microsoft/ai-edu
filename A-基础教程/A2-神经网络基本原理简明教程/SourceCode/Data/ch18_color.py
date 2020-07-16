# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from PIL import Image, ImageDraw, ImageColor
import numpy as np
import matplotlib.pyplot as plt


train_data_name = "../../data/ch18.train_color.npz"
test_data_name = "../../data/ch18.test_color.npz"

colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]

def circle(drawObj,r):
    x0 = np.random.randint(0,14)
    y0 = np.random.randint(0,14)
    x1 = np.random.randint(14,28)
    y1 = np.random.randint(14,28)
    color = colors[r]
    drawObj.ellipse([x0,y0,x1,y1], fill=color, outline=color)
    return r
    
def rectangle(drawObj,r):
    x0 = np.random.randint(0,14)
    y0 = np.random.randint(0,14)
    x1 = np.random.randint(14,28)
    y1 = np.random.randint(14,28)
    color = colors[r]
    drawObj.rectangle([x0,y0,x1,y1], fill=color, outline=color)
    return r

def triangle(drawObj,r):
    x0 = np.random.randint(0,14)
    y0 = np.random.randint(0,14)
    x1 = np.random.randint(14,28)
    y1 = np.random.randint(0,14)
    x2 = np.random.randint(0,14)
    y2 = np.random.randint(14,28)
    x3 = np.random.randint(14,28)
    y3 = np.random.randint(14,28)
    color = colors[r]
    if r == 0:
        drawObj.polygon([x0,y0,x1,y1,x2,y2], fill=color, outline=color)
    elif r == 1:
        drawObj.polygon([x0,y0,x1,y1,x3,y3], fill=color, outline=color)
    elif r == 2:
        drawObj.polygon([x0,y0,x2,y2,x3,y3], fill=color, outline=color)
    else:
        drawObj.polygon([x1,y1,x2,y2,x3,y3], fill=color, outline=color)
    return r

def line(drawObj,r):
    x0 = np.random.randint(0,28)
    y0 = np.random.randint(0,28)
    x1 = np.random.randint(0,28)
    y1 = np.random.randint(0,28)
    color = colors[r]
    drawObj.line([x0,y0,x1,y1], fill=color, width=2)
    return r

def diamond(drawObj,r):
    x0 = np.random.randint(0,14)
    y0 = np.random.randint(0,14)
    x1 = np.random.randint(14,28)
    y1 = np.random.randint(14,28)
    x2 = (x0+x1)/2
    y2 = (y0+y1)/2
    color = colors[r]
    drawObj.polygon([x2,y0,x1,y2,x2,y1,x0,y2], fill=color, outline=color)
    return r

def color():
    images_triangle = np.empty((1200,3,28,28))
    label_triangle = np.empty((1200,1))
    rand = np.random.randint(0,6,1200)
    for i in range(1200):
        img = Image.new("RGB", [28,28], "black")
        drawObj = ImageDraw.Draw(img)
        triangle(drawObj, rand[i])
        images_triangle[i] = np.array(img).transpose(2,0,1)
        #plt.imshow(images_triangle[i].transpose(1,2,0))
        #plt.show()
        label_triangle[i] = rand[i]
        
    images_circle = np.empty((1200,3,28,28))
    label_circle = np.empty((1200,1))
    rand = np.random.randint(0,6,1200)
    for i in range(1200):
        img = Image.new("RGB", [28,28], "black")
        drawObj = ImageDraw.Draw(img)
        circle(drawObj, rand[i])
        images_circle[i] = np.array(img).transpose(2,0,1)
        label_circle[i] = rand[i]

    images_rectangle = np.empty((1200,3,28,28))
    label_rectangle = np.empty((1200,1))
    rand = np.random.randint(0,6,1200)
    for i in range(1200):
        img = Image.new("RGB", [28,28], "black")
        drawObj = ImageDraw.Draw(img)
        rectangle(drawObj, rand[i])
        images_rectangle[i] = np.array(img).transpose(2,0,1)
        label_rectangle[i] = rand[i]

    images_line = np.empty((1200,3,28,28))
    label_line = np.empty((1200,1))
    rand = np.random.randint(0,6,1200)
    for i in range(1200):
        img = Image.new("RGB", [28,28], "black")
        drawObj = ImageDraw.Draw(img)
        line(drawObj, rand[i])
        images_line[i] = np.array(img).transpose(2,0,1)
        label_line[i] = rand[i]

    images_diamond = np.empty((1200,3,28,28))
    label_diamond = np.empty((1200,1))
    rand = np.random.randint(0,6,1200)
    for i in range(1200):
        img = Image.new("RGB", [28,28], "black")
        drawObj = ImageDraw.Draw(img)
        diamond(drawObj, rand[i])
        images_diamond[i] = np.array(img).transpose(2,0,1)
        label_diamond[i] = rand[i]

    images_train = np.empty((5000,3,28,28))
    label_train = np.empty((5000,1))

    images_train[0:1000] = images_circle[0:1000]
    label_train[0:1000] = label_circle[0:1000]
    
    images_train[1000:2000] = images_rectangle[0:1000]
    label_train[1000:2000] = label_rectangle[0:1000]
    images_train[2000:3000] = images_triangle[0:1000]
    label_train[2000:3000] = label_triangle[0:1000]
    images_train[3000:4000] = images_diamond[0:1000]
    label_train[3000:4000] = label_diamond[0:1000]
    images_train[4000:5000] = images_line[0:1000]
    label_train[4000:5000] = label_line[0:1000]


    images_test = np.empty((1000,3,28,28))
    label_test = np.empty((1000,1))
    
    images_test[0:200] = images_circle[1000:1200]
    label_test[0:200] = label_circle[1000:1200]
    images_test[200:400] = images_rectangle[1000:1200]
    label_test[200:400] = label_rectangle[1000:1200]
    images_test[400:600] = images_triangle[1000:1200]
    label_test[400:600] = label_triangle[1000:1200]
    images_test[600:800] = images_diamond[1000:1200]
    label_test[600:800] = label_diamond[1000:1200]
    images_test[800:1000] = images_line[1000:1200]
    label_test[800:1000] = label_line[1000:1200]

    seed = np.random.randint(0,100)
    np.random.seed(seed)
    np.random.shuffle(images_train)
    np.random.seed(seed)
    np.random.shuffle(label_train)
    np.random.seed(seed)
    np.random.shuffle(images_test)
    np.random.seed(seed)
    np.random.shuffle(label_test)

    np.savez(train_data_name, data=images_train, label=label_train)
    np.savez(test_data_name, data=images_test, label=label_test)


if __name__ == '__main__':
    color()
