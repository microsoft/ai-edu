# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from PIL import Image, ImageDraw
import numpy as np

train_data_name = "../../data/ch18.train_shape.npz"
test_data_name = "../../data/ch18.test_shape.npz"

def circle(drawObj):
    x0 = np.random.randint(0,14)
    y0 = np.random.randint(0,14)
    x1 = np.random.randint(14,28)
    y1 = np.random.randint(14,28)
    drawObj.ellipse([x0,y0,x1,y1], fill='white', outline='white')
    
def rectangle(drawObj):
    x0 = np.random.randint(0,14)
    y0 = np.random.randint(0,14)
    x1 = np.random.randint(14,28)
    y1 = np.random.randint(14,28)
    drawObj.rectangle([x0,y0,x1,y1], fill='white', outline='white')

def triangle(drawObj):
    x0 = np.random.randint(0,14)
    y0 = np.random.randint(0,14)
    x1 = np.random.randint(14,28)
    y1 = np.random.randint(0,14)
    x2 = np.random.randint(0,14)
    y2 = np.random.randint(14,28)
    x3 = np.random.randint(14,28)
    y3 = np.random.randint(14,28)
    r = np.random.randint(0,3)
    if r == 0:
        drawObj.polygon([x0,y0,x1,y1,x2,y2], fill='white', outline='white')
    elif r == 1:
        drawObj.polygon([x0,y0,x1,y1,x3,y3], fill='white', outline='white')
    elif r == 2:
        drawObj.polygon([x0,y0,x2,y2,x3,y3], fill='white', outline='white')
    else:
        drawObj.polygon([x1,y1,x2,y2,x3,y3], fill='white', outline='white')

def line(drawObj):
    x0 = np.random.randint(0,28)
    y0 = np.random.randint(0,28)
    x1 = np.random.randint(0,28)
    y1 = np.random.randint(0,28)
    drawObj.line([x0,y0,x1,y1], fill='white', width=2)
   

def diamond(drawObj):
    x0 = np.random.randint(0,14)
    y0 = np.random.randint(0,14)
    x1 = np.random.randint(14,28)
    y1 = np.random.randint(14,28)
    x2 = (x0+x1)/2
    y2 = (y0+y1)/2
    drawObj.polygon([x2,y0,x1,y2,x2,y1,x0,y2], fill='white', outline='white')

if __name__ == '__main__':
    images_triangle = np.empty((1200,1,28,28))
    for i in range(1200):
        img = Image.new("L", [28,28], "black")
        drawObj = ImageDraw.Draw(img)
        triangle(drawObj)
        images_triangle[i,0] = np.array(img)
        
    images_circle = np.empty((1200,1,28,28))
    for i in range(1200):
        img = Image.new("L", [28,28], "black")
        drawObj = ImageDraw.Draw(img)
        circle(drawObj)
        images_circle[i,0] = np.array(img)

    images_rectangle = np.empty((1200,1,28,28))
    for i in range(1200):
        img = Image.new("L", [28,28], "black")
        drawObj = ImageDraw.Draw(img)
        rectangle(drawObj)
        images_rectangle[i,0] = np.array(img)

    images_line = np.empty((1200,1,28,28))
    for i in range(1200):
        img = Image.new("L", [28,28], "black")
        drawObj = ImageDraw.Draw(img)
        line(drawObj)
        images_line[i,0] = np.array(img)

    images_diamond = np.empty((1200,1,28,28))
    for i in range(1200):
        img = Image.new("L", [28,28], "black")
        drawObj = ImageDraw.Draw(img)
        diamond(drawObj)
        images_diamond[i,0] = np.array(img)
    

    images_train = np.empty((5000,1,28,28))
    label_train = np.empty((5000,1))
    images_train[0:1000] = images_circle[0:1000]
    label_train[0:1000] = 0
    images_train[1000:2000] = images_rectangle[0:1000]
    label_train[1000:2000] = 1
    images_train[2000:3000] = images_triangle[0:1000]
    label_train[2000:3000] = 2
    images_train[3000:4000] = images_diamond[0:1000]
    label_train[3000:4000] = 3
    images_train[4000:5000] = images_line[0:1000]
    label_train[4000:5000] = 4


    images_test = np.empty((1000,1,28,28))
    label_test = np.empty((1000,1))
    images_test[0:200] = images_circle[1000:1200]
    label_test[0:200] = 0
    images_test[200:400] = images_rectangle[1000:1200]
    label_test[200:400] = 1
    images_test[400:600] = images_triangle[1000:1200]
    label_test[400:600] = 2
    images_test[600:800] = images_diamond[1000:1200]
    label_test[600:800] = 3
    images_test[800:1000] = images_line[1000:1200]
    label_test[800:1000] = 4

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
