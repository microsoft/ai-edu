# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import os
import json


def get_repo(url_adder):
    dir = 'data/py/repos'
    os.makedirs(dir, exist_ok=True)
    try:
        if url_adder.startswith('https://github.com/'):
            url_adder = url_adder[len('https://github.com/'):]
        else:
            url_adder = url_adder.strip('/')
        print('getting', url_adder)

        if os.name == 'nt':
          batchFile = os.path.normcase(os.path.join(os.path.dirname(__file__), 'getcode_with_full_git_path.bat'))
          command = batchFile + ' ' + url_adder.split('/')[0] + ' ' + url_adder.split('/')[1] + ' ' + os.path.normcase(dir)
          print(command)
          os.system(command)

        else:
          batchFile = os.path.normcase(os.path.join(os.path.dirname(__file__), 'getcode_with_full_git_path.sh'))
          command = 'bash ' + batchFile + ' ' + url_adder.split('/')[0] + ' ' + url_adder.split('/')[1] + ' ' + dir
          print(command)
          os.system(command)
    except:
        print("ERROR WHEN GET", url_adder)


def get_data(ans, dir_name):
    files = os.listdir(dir_name)
    for file in files:
        if file.endswith('.json'):
            with open(dir_name + r'/' + file) as f:
                data = json.load(f)['data']
                for url in data:
                    ans.add(url['url'])
    return ans


if __name__ == '__main__':
    ans = set()
    ans = get_data(ans, 'data/py/repoList')
    for i in ans:
        get_repo(i)
        
