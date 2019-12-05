from PIL import Image, ImageDraw, ImageFont, ImageFile
import matplotlib.pyplot as plt
import numpy as np
import PIL


class Synthesizor:
    def __init__(self, size_rate=1):
        logo_size_rate = 1.5
        self.QRcode = Image.open('imgs/qrcode.jpg')
        self.QRcode_size = (int(35 * size_rate * logo_size_rate), int(35 * size_rate * logo_size_rate))
        self.QRcode = self.QRcode.resize(self.QRcode_size, Image.ANTIALIAS)
        self.wechat = Image.open('imgs/wechat.jpg')
        self.wechat_size = (int(7 * size_rate * logo_size_rate), int(7 * size_rate * logo_size_rate))
        self.wechat = self.wechat.resize(self.wechat_size, Image.ANTIALIAS)
        self.logo = Image.open('imgs/logo.png')
        self.logo_size = (int(15 * size_rate * logo_size_rate), int(15 * size_rate * logo_size_rate))
        self.logo = self.logo.resize(self.logo_size, Image.ANTIALIAS)
        self.default_syn_h = 465
        self.default_syn_w = 288
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        self.syn_h = int(self.default_syn_h * size_rate)
        self.syn_w = int(self.default_syn_w * size_rate)
        self.s_sample_rate = 0.05

        self.x_border = int(5 * size_rate)
        self.y_border = int(7 * size_rate)
        self.font_color = (237, 0, 32)
        self.max_font_size = int(30 * size_rate)

        self.text0_left_h = int((180 + 39) * size_rate)
        self.text1_left_h = int((180) * size_rate)
        self.text0_left_v = int((345 + 39) * size_rate)
        self.text1_left_v = int((345) * size_rate)
        self.text_up = int(41 * size_rate)

        self.info = ["小程序", "景", "情"]
        self.info_font_size = [int(7 * size_rate * logo_size_rate), int(10 * size_rate * logo_size_rate)]
        self.info_margin = [int(1 * size_rate * logo_size_rate), int(3 * size_rate * logo_size_rate)]
        self.info_color = (128, 128, 128)
        self.wechat_mask_value = 128
        self.QRcode_mask_value = 128
        self.logo_mask_value = 128
        self.wechat_loc_h = {"horizontal": (int((10) * size_rate), int((10) * size_rate)),
                             "vertical": (int((10) * size_rate), int((10) * size_rate))}
        self.wechat_loc_v = {"horizontal": (int((431-10-70 * logo_size_rate) * size_rate), int((465-10-30 * logo_size_rate) * size_rate)),
                             "vertical": (int((431 - 10 - 30 * logo_size_rate) * size_rate), int((465 - 10 - 70 * logo_size_rate) * size_rate))}


    def logistic(self, x, alpha=1):
        return 1 / (np.exp(-alpha * x) + 1)

    def linmap(self, x, ori_range, map_range):
        return (x - ori_range[0]) / float(ori_range[1] - ori_range[0]) * (map_range[1] - map_range[0]) + map_range[0]

    def powmap(self, x, ori_range, map_range, pow):
        return ((x - ori_range[0]) / float(ori_range[1] - ori_range[0])) ** pow * (map_range[1] - map_range[0]) + map_range[0]

    def calcHW(self, h, w):
        if 4*h > 5*w:
            type = "vertical"
            def_h = self.syn_h
            def_w = int(self.syn_w * 1.1)
        else:
            type = "horizontal"
            def_h = self.syn_h // 2
            def_w = self.syn_w
        return type, def_h, def_w

    def gen_mask(self, type="horizontal", fill=False, shape=None):

        if shape is None:
            height = self.syn_h
            width = self.syn_w
        else:
            height = shape[0]
            width = shape[1]
        # import os
        # mask_dir = "imgs/masks"
        # if not os.path.exists(mask_dir):
        #     os.makedirs(mask_dir)
        # mask_file_name = "imgs/masks/type@{}_height@{}_width@{}.npy".format(type, height, width)
        # if os.path.exists(mask_file_name):
        #     mask = np.load(mask_file_name)
        #     return mask

        if type == "horizontal":
            if fill:
                cut_point, loose, tight, pow, biasleft, biasright, biaspow = (width, 30, 50, 2.5, 0.6, 0.5, 2.5)
            else:
                cut_point, loose, tight, pow, biasleft, biasright, biaspow = (width, 15, 30, 2.5, 0.68, 0.6, 2.5)
            x = np.ones((1, width))
            y = np.linspace(0, 1, height, False).reshape((height, 1))
            y = self.logistic(0.5-y, alpha=30)
            mask = np.matmul(y, x)
            y = np.linspace(0, 1, height, False)
            for i in range(0, cut_point):
                mask[:, i] = self.logistic(self.powmap(i, [0,cut_point], [biasleft,biasright], biaspow)-y, alpha=self.powmap(i, [0,cut_point], [loose,tight], pow))
            mask = np.stack([mask, mask, mask], axis=2)
            # plt.imshow(mask)
            # plt.show()
        else:
            assert type == "vertical"
            if fill:
                cut_point, loose, tight, pow, biasleft, biasright, biaspow = (height, 30, 50, 2.5, 0.6, 0.5, 2.5)
            else:
                cut_point, loose, tight, pow, biasleft, biasright, biaspow = (height, 30, 12, 2.5, 0.68, 0.6, 2.5)
            mask = np.zeros((height, width))
            x = np.linspace(0, 1, width, False)
            for i in range(0, cut_point):
                mask[i, :] = self.logistic(x - self.powmap(i, [0,cut_point], [biasleft,biasright], biaspow), alpha=self.powmap(i, [0,cut_point], [loose,tight], pow))
            mask = np.stack([mask, mask, mask], axis=2)
        # np.save(mask_file_name, mask)
        return mask

    def draw_text(self, im1, strs, x, y, font, font_color=None, margin=None, h=None, w=None, with_box=True, direction="vertical"):

        if not isinstance(im1, Image.Image):
            im1 = Image.fromarray(np.uint8(im1 * 255))

        draw = ImageDraw.Draw(im1)
        print(font.getsize(strs))
        right = 0
        down = 0
        if margin is None:
            margin = 6
        print(strs)
        if font_color is None:
            font_color = self.font_color
        if direction == "vertical":
            for k, s2 in enumerate(strs):
                if k == 0:
                    if h is None:
                        w, h = font.getsize(s2)
                    else:
                        w, _ = font.getsize(s2)
                else:
                    down = down + h + margin
                draw.text((x + right, y + down), s2, font_color, font=font)
            if with_box:
                bbox = [x - self.x_border, y - self.y_border, x + w + self.x_border, y + down + h + self.y_border]
                draw.rectangle(bbox, outline=font_color, width=1)
        else:
            assert direction == "horizontal"
            for k, s2 in enumerate(strs):
                if k == 0:
                    if w is None:
                        w, h = font.getsize(s2)
                    else:
                        _, h = font.getsize(s2)
                else:
                    right = right + w + margin
                draw.text((x + right, y + down), s2, font_color, font=font)

        return im1

    def test_text(self, strs, font):
        w, h = font.getsize(strs[0])
        return h

    def apply_bias(self, a, b):
        return [a[i] + b[i] for i in range(len(a))]

    def generate_logo(self, im, loc, dir="horizontal"):
        if dir == "horizontal":
            b_qrcode = [self.QRcode_size[0] + self.info_margin[0], 0]
            b_wechat = [self.wechat_size[0] + self.info_margin[0], 0]
            b_change_line = [0, self.wechat_size[1] + self.info_margin[1]]
            b_one_ch = [self.info_font_size[1] + self.info_margin[0], 0]
            b_logo = [self.logo_size[0] + self.info_margin[0], 0]
            b_logo_txt_adj = [0, (self.logo_size[1] - self.info_font_size[1])//2]
            QRcode_loc = loc["horizontal"]
            wechat_loc = self.apply_bias(QRcode_loc, b_qrcode)
            txt1_loc = self.apply_bias(wechat_loc, b_wechat)
            txt2_loc = self.apply_bias(wechat_loc, b_change_line)
            logo2_loc = self.apply_bias(txt2_loc, b_one_ch)
            txt3_loc = self.apply_bias(logo2_loc, b_logo)
            txt2_loc = self.apply_bias(txt2_loc, b_logo_txt_adj)
            txt3_loc = self.apply_bias(txt3_loc, b_logo_txt_adj)
            info_margin = self.info_margin
        else:
            assert dir == "vertical"
            b_qrcode = [0, self.QRcode_size[0] + self.info_margin[0]]
            b_wechat = [0, self.wechat_size[0] + self.info_margin[0]]
            b_change_line = [self.wechat_size[1] + self.info_margin[1], 0]
            b_one_ch = [0, self.info_font_size[1] + self.info_margin[0]]
            b_logo = [0, self.logo_size[0] + self.info_margin[0]]
            b_logo_txt_adj = [(self.logo_size[1] - self.info_font_size[1]) // 2, 0]
            QRcode_loc = loc["vertical"]
            wechat_loc = self.apply_bias(QRcode_loc, b_qrcode)
            txt1_loc = self.apply_bias(wechat_loc, b_wechat)
            txt2_loc = self.apply_bias(wechat_loc, b_change_line)
            logo2_loc = self.apply_bias(txt2_loc, b_one_ch)
            txt3_loc = self.apply_bias(logo2_loc, b_logo)
            txt2_loc = self.apply_bias(txt2_loc, b_logo_txt_adj)
            txt3_loc = self.apply_bias(txt3_loc, b_logo_txt_adj)
            info_margin = [self.info_margin[1], self.info_margin[0]]
        font_names = [u"imgs/simsun.ttc", u"imgs/AdobeFangsongStd-Regular.otf"]
        info_font = [ImageFont.truetype(font_names[i], size=int(self.info_font_size[i])) for i in range(len(self.info_font_size))]

        # im = self.draw_text(im, self.info[0], txt1_loc[0], txt1_loc[1], info_font[0],
        #                         font_color=self.info_color, margin=info_margin[0], direction=dir,
        #                         with_box=False)
        im = self.draw_text(im, self.info[1], txt2_loc[0], txt2_loc[1], info_font[1],
                                font_color=self.info_color, margin=info_margin[0], direction=dir,
                                with_box=False)
        im = self.draw_text(im, self.info[2], txt3_loc[0], txt3_loc[1], info_font[1],
                                font_color=self.info_color, margin=info_margin[0], direction=dir,
                                with_box=False)
        # im.paste(self.wechat, wechat_loc, mask=Image.fromarray(np.uint8(np.ones(self.wechat_size)*self.wechat_mask_value)))
        im.paste(self.QRcode, QRcode_loc, mask=Image.fromarray(np.uint8(np.ones(self.QRcode_size)*self.QRcode_mask_value)))
        im.paste(self.logo, logo2_loc, mask=Image.fromarray(np.uint8(np.ones(self.logo_size)*self.logo_mask_value)))
        return im

    def generate_image(self, user_image_path="imgs/download.jpg", poeitcs=["人饮春节酒", "马渡风月关"], fill=False):

        im = Image.open(user_image_path)
        w, h = im.size
        type, def_h, def_w = self.calcHW(h, w)
        # print(type, def_h, def_w)
        ratio = max(def_h / h, def_w / w)  # Guarantee no white space
        im = im.resize((int(w * ratio), int(h * ratio)), Image.ANTIALIAS)

        image = np.asarray(im)
        image = image / 255
        h, w, c = np.shape(image)
        # print(h, w, c)
        upleft = (h // 2 - def_h // 2, w // 2 - def_w // 2)
        subimg = image[upleft[0]:upleft[0] + def_h, upleft[1]:upleft[1] + def_w, :]
        if type == "horizontal":
            m = np.mean(subimg[:int(self.s_sample_rate * self.syn_h), :], (0, 1))
            sub_white = np.ones((self.syn_h - def_h, self.syn_w, c))
            sub_white[:, :, 0] *= m[0]
            sub_white[:, :, 1] *= m[1]
            sub_white[:, :, 2] *= m[2]
            image = np.concatenate([sub_white, subimg], axis=0)
        else:
            assert type == "vertical"
            m = np.mean(subimg[:, -int(self.s_sample_rate * self.syn_w):], (1))
            m = np.stack([m for i in range(int(0.4*self.syn_w))], axis=1)
            sub_white = np.ones((self.syn_h, int(0.4*self.syn_w), c))
            sub_white *= m
            image = np.concatenate([subimg, sub_white], axis=1)
        mask = self.gen_mask(type, fill, image.shape)
        white = np.ones((image.shape[0], image.shape[1], c))
        result = white * mask + image * (1-mask)

        word_num = max([len(poeitcs[i]) for i in range(len(poeitcs))])
        print(word_num)
        font_size = min(self.max_font_size, int(self.max_font_size * 5 / word_num))
        word_font = ImageFont.truetype(u"imgs/AdobeFangsongStd-Regular.otf", size=int(font_size))
        word_height = max([self.test_text(poeitcs[i], word_font) for i in range(len(poeitcs))])
        if type == "horizontal":
            result = self.draw_text(result, poeitcs[0], self.text0_left_h, self.text_up, word_font, h=word_height)
            result = self.draw_text(result, poeitcs[1], self.text1_left_h, self.text_up, word_font, h=word_height)
            result = self.generate_logo(result, self.wechat_loc_h, dir=type)
        else:
            assert type == "vertical"
            result = self.draw_text(result, poeitcs[0], self.text0_left_v, self.text_up, word_font, h=word_height)
            result = self.draw_text(result, poeitcs[1], self.text1_left_v, self.text_up, word_font, h=word_height)
            result = self.generate_logo(result, self.wechat_loc_v, dir=type)
        # result = Image.fromarray(np.uint8(result * 255))
        return result


def generate_final_image(user_image_path="imgs/vertical3.jpg", background_image_path="./bg.jpg", poeitcs=["长风破浪会有时", "直挂云帆济沧海"], size_rate=2):
    return Synthesizor(size_rate).generate_image(user_image_path,poeitcs=poeitcs,fill=False)


if __name__ == '__main__':
    import time
    start = time.time()
    result = generate_final_image(size_rate=6)
    print("Size_rage-{}: Elapsed {} seconds".format(6, time.time() - start))
    result.save('imgs/3-1.jpg')