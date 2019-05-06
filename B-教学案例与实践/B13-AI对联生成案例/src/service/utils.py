# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# coding=utf-8
import time
import json
# from requests_toolbelt import MultipartDecoder
from multipart_decoder import MultipartDecoderV2 as MultipartDecoder
from PIL import Image
from io import BytesIO
import os
import pickle
import shutil
from cognitive_service import call_cv_api
from word_matching import find_shanglian
from synthesis_2 import generate_final_image
import os

base_dir = os.path.dirname(os.path.abspath(__file__))


def generate_id():
    return int(time.time() * 1000)


def do_upload_image(up_to_down_model, shanglian_file_content, dict_1, dict_2, raw_data):
    metadata = raw_data
    id = generate_id()
    bdry = metadata.split(b"\n")[0][2:]
    decoder = MultipartDecoder(metadata, b'multipart/form-data;boundary=' + bdry)
    img_bytes = decoder.parts[1].content
    # image = Image.open(BytesIO(decoder.parts[1].content))
    if not os.path.exists('raw_images'):
        os.mkdir('raw_images')
    with open('raw_images/{}.jpg'.format(id), 'wb') as f:
        f.write(img_bytes)

    # Todo: pass image to backend to get poetry
    t0 = time.time()
    rs_result = call_cv_api(decoder.parts[1].content)
    t1 = time.time()


    print('load file time:', time.time() - t1)
    shanglian_results = find_shanglian(rs_result, shanglian_file_content, dict_1, dict_2, final_output_number=3)
    t2 = time.time()
    #final_results = []
    #for shanglian in shanglian_results:
    #    shanglian_modify = ""
    #    for i in shanglian:
    #        shanglian_modify = shanglian_modify + i + ' '
    #    shanglian_modify = shanglian_modify[:-1]
    #    xialian = up_to_down_model.get_next(u'{}'.format(shanglian_modify))
    #    result = str(shanglian + '，' + xialian)  # 中文逗号会出问题吗？
    #    final_results.append(result)
    final_results = up_to_down_model.get_next(",".join(shanglian_results))
    final_results = [ shanglian_results[i] + ',' + final_results[i] for i in range(len(final_results)) ]
    t3 = time.time()
    print('1: {}, 2: {}, 3: {}'.format(t1-t0, t2-t1, t3-t2))
    '''
    form = {'id': id,
            'poetry1': u"花开花落，云卷云舒",
            'poetry2': u"床前明月光，疑是地上霜",
            'poetry3': u"苟利国家生死以， 岂因祸福避趋之"}
    '''
    form = {'id': id,
            'poetry1': u"{}".format(final_results[0]),
            'poetry2': u"{}".format(final_results[1]),
            'poetry3': u"{}".format(final_results[2])}
    return json.dumps(form)

def do_modify_poetry(up_to_down_model, metadata):
    id = metadata['id']
    shanglian, xialian = metadata['poetry'].split(',')
    new_xialian = up_to_down_model.get_next(shanglian)
    new_poetry = '{},{}'.format(shanglian, new_xialian)
    form = {'id': id,
            'poetry': new_poetry}
    return json.dumps(form)

def do_selected_poetry(metadata):
    id = metadata['id']
    choice = metadata['choice']

    # Todo: pass image and selected poetry to get synthesis result
    poetics = choice.split(',')
    print(poetics)
    syn_image_content = generate_final_image(user_image_path=os.path.join(base_dir, 'raw_images/{}.jpg'.format(id)),
                                             background_image_path=os.path.join(base_dir, 'imgs/background.jpg'),
                                             poeitcs=poetics)
    if not os.path.exists(os.path.join(base_dir, 'syn_images')):
        os.mkdir(os.path.join(base_dir, 'syn_images'))
    syn_image_content.save(os.path.join(base_dir, 'syn_images/{}_syn.jpg'.format(id)), 'jpeg')
    # shutil.copy('raw_images/{}.jpg'.format(id), 'syn_images/{}_syn.jpg'.format(id))

    # base_url = 'http://poeticimage.eastus.cloudapp.azure.com:8080'
    # base_url = 'https://poeticimage.imdo.co'
    # base_url = 'https://poeticimage.not-lug.lug.ustc.edu.cn:8000'
    # img_url = '{}/syn_images/{}_syn.jpg'.format(base_url, id)
    img_url = '{}_syn.jpg'.format(id)

    form = {'id': id,
            'url': img_url}
    return json.dumps(form)


def do_comment(metadata):
    # Todo: upload comments to database
    id = metadata['id']
    if not os.path.exists('comment'):
        os.mkdir('comment')
    with open('comment/{}.json'.format(id), 'w') as f:
        json.dump(metadata, f)
    import pprint
    pprint.pprint(metadata)
    return json.dumps(metadata)
