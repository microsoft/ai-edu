# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from http.server import BaseHTTPRequestHandler, HTTPServer, SimpleHTTPRequestHandler
from io import BytesIO
from utils import *
import chardet
import ssl
from socketserver import ThreadingMixIn

import os
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_model")
t2t_dir = os.path.join(model_path, "tensor2tensor-1.2.9")
t2t_main = os.path.join(t2t_dir, "tensor2tensor")
t2t_bin = os.path.join(t2t_main, "bin")
import sys
sys.path.append(model_path)
sys.path.append(t2t_dir)
sys.path.append(t2t_main)
sys.path.append(t2t_bin)


import up2down_class
import threading


class PiHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super(PiHTTPRequestHandler, self).__init__(*args, **kwargs)
        self.locked_list = [False, False]

    #def do_GET(self):
    #    self.send_response(200)
    #    self.end_headers()
    #    self.wfile.write(b'Hello, world!')

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        # print(chardet.detect(self.rfile.read(content_length)))
        metadata = self.rfile.read(content_length)
        # datetime = time.strftime('%Y-%m-%d-%H-%M-%S')
        # if not os.path.exists('tmp'):
        #     os.mkdir('tmp')
        # with open(os.path.join('tmp', datetime), 'w') as f:
        #     f.write(metadata)
        self.send_response(200)
        self.end_headers()
        message = self.process_metadata(metadata)
        response = BytesIO()
        response.write(message.encode())
        self.wfile.write(response.getvalue())

    def process_metadata(self, metadata):
        # global generate_model_0, generate_model_1
        # global lock_0, lock_1
        # global index
        # global model_num
        # global lock_id
        if metadata[:2] == b'--':
            # temp_index = 0
            # if lock_id.acquire():
            #     index += 1
            #     index = index % model_num
            #     temp_index = index
            #     lock_id.release()
            # if eval("lock_{}".format(temp_index)).acquire():
            #     print("use generate_model_{}".format(temp_index))
            #     message = do_upload_image(eval("generate_model_{}".format(temp_index)), metadata)
            #     eval("lock_{}".format(temp_index)).release()
            global generate_model_0
            global shanglian_file_content
            global dict_1
            global dict_2
            # message = do_upload_image(generate_model_0, metadata)
            message = do_upload_image(generate_model_0, shanglian_file_content, dict_1, dict_2, metadata)
        else:
            metadata = metadata.decode()
            metadata = json.loads(metadata)
            if metadata['type'] == 'select':
                message = do_selected_poetry(metadata)
            elif metadata['type'] == 'comment':
                message = do_comment(metadata)
            elif metadata['type'] == 'modify':
                message = do_modify_poetry(generate_model_0, metadata)
            else:
                message = 'hi'
        return message


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""


def run(server_class=HTTPServer, handler_class=PiHTTPRequestHandler, port=80):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print('Starting httpd...')

    httpd.serve_forever()


if __name__ == "__main__":
    from sys import argv
    # lock_id = threading.Lock()
    # index = 0
    # model_num = 2
    # lock_0 = threading.Lock()
    # lock_1 = threading.Lock()
    # lock_2 = threading.Lock()
    # lock_3 = threading.Lock()
    generate_model_0 = up2down_class.up2down_class()
    shanglian_filename = os.path.join(base_dir, 'train.txt.up')
    dict_1_filename = os.path.join(base_dir, 'dict_1.txt')
    dicr_2_filename = os.path.join(base_dir, 'dict_2.txt')
    with open(shanglian_filename, 'r', encoding='utf-8') as in_file:
        shanglian_file_content = in_file.readlines()
    dict_1 = pickle.load(open(dict_1_filename, "rb"))
    dict_2 = pickle.load(open(dicr_2_filename, "rb"))

    # generate_model_1 = up2down_class.up2down_class()
    # generate_model_2 = up2down_class.up2down_class()
    # generate_model_3 = up2down_class.up2down_class()
    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run(port=8000)

