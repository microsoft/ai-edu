# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# coding=utf-8
from PIL import Image, ImageDraw, ImageFont, ImageFile
import matplotlib.pyplot as plt
import numpy as np
import PIL

# im = Image.open(r"imgs/background.jpg")
# draw = ImageDraw.Draw(im)
# myfont = ImageFont.truetype(u"D:\\ASE\\Poetic-Image\\imgs\\font1556\\handwriting.ttf", size=35)
# fillcolor = 'black'
# draw.text((350, 400), u"还有谁！！", font=myfont, fill=fillcolor)
# plt.figure('abc')
# plt.imshow(im)
# plt.show()

ImageFile.LOAD_TRUNCATED_IMAGES = True

def gen_cir_mask(w, h):
    m_x, m_y = np.meshgrid(np.arange(w), np.arange(h))
    c_x = (w-1) / 2.0
    c_y = (h-1) / 2.0
    soft_mask = (m_x - c_x) ** 2 / (c_x ** 2) + (m_y - c_y) ** 2 / (c_y ** 2)
    Z = np.where(soft_mask<0.95, np.ones_like(soft_mask), np.zeros_like(soft_mask))
    Z_2 = np.where(abs(soft_mask-0.975)<0.025, np.ones_like(soft_mask), np.zeros_like(soft_mask))
    #Z_2 = np.where((1-soft_mask)<0.05, np.ones_like(soft_mask), np.zeros_like(soft_mask))
    return Z, Z_2
def generate_final_image(user_image_path="imgs/download.jpg", background_image_path="./bg.jpg", poeitcs=["Sample Text 1", "Sample Text 1"]):

    im = Image.open(user_image_path)
    h, w = im.size
    ratio = min(800./h, 600./w)
    im = im.resize((int(h * ratio), int(w * ratio)), Image.ANTIALIAS)
    image = np.asarray(im)
    h, w, c= np.shape(image)
    #print(h,w,c,width, height)

    bg_img_2 =  Image.open(background_image_path) # background image
    bg_img_2_resized = bg_img_2.resize((800,600), Image.ANTIALIAS)
    bg_crop= np.asarray(np.asarray(bg_img_2_resized))[0:h, 0:w,:]

    mask,mask_2 = gen_cir_mask(w, h)
    mask =  np.transpose(np.repeat(np.expand_dims(mask,0), 3, axis=0), (1,2,0))
    mask_2 =  np.transpose(np.repeat(np.expand_dims(mask_2,0), 3, axis=0), (1,2,0))

    #word_font = ImageFont.truetype("arial.ttf", 40)
    word_font = ImageFont.truetype(u"/home/administor/code/Poetic-Image/imgs/handwriting.ttf", size=35)

    # generate mask
    image = np.asarray(im)
    image_2 = np.where(mask==1, image,bg_crop)
    image_2 = np.where(mask_2==1, 0, image_2)

    im_2 = Image.fromarray(np.uint8(image_2))
    im_resized = im_2.resize((500,300), Image.ANTIALIAS)
    pic_height, pic_width,_ = np.shape(im_resized)
    bg_height, bg_width,_ = np.shape(bg_img_2_resized)
    top_left_h = int(bg_height*0.7/2-pic_height/2)
    top_left_w = int(bg_width/2 - pic_width/2)
    bg_img_2_resized.paste(im_resized, (top_left_w,top_left_h))

    draw_whole = ImageDraw.Draw(bg_img_2_resized)
    draw_whole.text((300, 400), poeitcs[0], (0,0,0), font=word_font)
    draw_whole.text((300, 500), poeitcs[1], (0,0,0),font=word_font)
    #plt.imshow(bg_img_2_resized)

    #bg_img_2_resized.save(output_path, 'jpeg')
    return bg_img_2_resized

if __name__ == "__main__":
    final_output_image = generate_final_image()
