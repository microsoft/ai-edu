# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# coding=utf-8
import requests
import json
from io import BytesIO


class Client:
    def __init__(self, port=8000, url="http://localhost"):
        self.name = "test_client"
        self.port = port
        self.url = url

    def upload_formdata(self, formdata_file_path):
        with open(formdata_file_path, 'rb') as f:
            formdata = f.read()
            r = requests.post("{0}:{1}".format(self.url, self.port), data=formdata, verify=False)
            return r.text

    def upload_json(self, upload_dict):
        json_dumps = json.dumps(upload_dict)
        r = requests.post("{0}:{1}".format(self.url, self.port), data=json_dumps, verify=False)
        return r.text
    # def img2tag(self, img_path):
    #     # image = Image.open()
    #     # plt.imshow(image)
    #     with open(img_path, "rb") as f:
    #         form = {'type': 'imgs',
    #                 # 'data': f.read().encode('base64')
    #                 }
    #         form = json.dumps(form)
    #         r = requests.post("{0}:{1}".format(self.url, self.port), data=form)
    #     tags = json.loads(r.text)
    #     return tags


if __name__ == '__main__':
    client = Client(url="http://127.0.0.1", port=8000)
    # print(client.img2tag("imgs/download.jpg"))
    import time
    start = time.time()
    print(client.upload_formdata('test_formdata'))
    print('time: {}s'.format(time.time() - start))
    #upload_dict = {'type': 'select',
    #               'id': 1544343434522,
    #               'choice': u'流水高山云动影，清风明月柳扬眉'}
    #print(client.upload_json(upload_dict))
