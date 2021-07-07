# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import requests
from bs4 import BeautifulSoup
import json
import time
import random
import os

base_url = '/search?l=Python&o=desc&q=python+license%3Amit&s=stars&type=Repositories'
thres = 3
base = 'https://github.com'
all_data = []
s = requests.Session()

s.headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0',
             'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
             'Accept-Encoding':'gzip, deflate, br',
             'Accept-Language':'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3'}
sleep_time = 10
dir_name = 'data/py/repoList'
os.makedirs(dir_name, exist_ok=True)


def get_data(url, page=1):
    global sleep_time
    while True:
        print(url, page)
        raw = s.get(base + url)
        ctx = raw.content
        if str(raw.status_code) == '200':
            break
        else:
            print('status code', raw.status_code)
            if sleep_time < 100:
                sleep_time += 2
            time.sleep(sleep_time)
            print('sleeping', sleep_time)
    soup = BeautifulSoup(ctx, 'html.parser')
    result_div = soup.find('ul', {'class': 'repo-list'})
    repos = result_div.find_all('li')
    repo_this_page = []
    need_next = True
    for repo in repos:
        url_adder = repo.find('a').attrs['href']
        star = repo.find_all('a', {'class': 'muted-link'})[-1].text.strip()
        if ('k' in star) or int(star) > thres:
            repo_this_page.append({'url': url_adder, 'star': star})
            all_data.append(url_adder)
        else:
            need_next = False
            break
    with open(dir_name + '/' + str(page) + '.json', 'w') as f:
        json.dump({'data': repo_this_page, 'page': page, 'url': url}, f)
    if need_next:
        next_page = soup.find('a', {'class': 'next_page'})
        time.sleep(random.randrange(4000, 15000) / 1000)
        try:
            next_href = next_page.attrs['href']
            get_data(next_href, page + 1)
        except:
            pass


if __name__ == '__main__':
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    get_data(base_url,1)