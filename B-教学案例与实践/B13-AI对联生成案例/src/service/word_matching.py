# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# -*- coding:utf-8 -*-
import time
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import sys
import json
import requests
import synonyms
from cognitive_service import call_cv_api, analysis_to_tags # 上传到服务器时需要修改
import os
import pickle
import random
base_dir = os.path.dirname(os.path.abspath(__file__))


def translate(word):
    # 有道词典 api
    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=null'
    # 传输的参数，其中 i 为需要翻译的内容
    key = {
        'type': "AUTO",
        'i': word,
        "doctype": "json",
        "version": "2.1",
        #'from': 'AUTO',
        #'to': 'Chinese',
        "keyfrom": "fanyi.web",
        "ue": "UTF-8",
        "action": "FY_BY_CLICKBUTTON",
        "typoResult": "true"
    }
    # key 这个字典为发送给有道词典服务器的内容
    response = requests.post(url, data=key)
    # 判断服务器是否相应成功
    if response.status_code == 200:
        return response.text
    else:
        #print("有道词典调用失败")
        return None

def get_reuslt(repsonse):
    result = json.loads(repsonse)
    return result['translateResult'][0][0]['tgt']

def find_shanglian(input_info, shanglian_file_content, dict_1, dict_2, final_output_number=5, random_tags=True):
    new_all_lines = shanglian_file_content
    start = time.clock()
    filename = os.path.join(base_dir, 'couplet_100k.txt')
    new_file_name = os.path.join(base_dir, 'train.txt.up')
    input_tag_list = ['moon']
    input_tag_list = analysis_to_tags(input_info)
    print("cognitive service tags are ")
    print(input_tag_list)
    list_trans = []
    final_input_tag_list = []
    translate_start = time.clock()
    if os.path.exists(os.path.join(base_dir, 'en2cn_dict.txt')):
        en2cn_dict = pickle.load(open(os.path.join(base_dir, 'en2cn_dict.txt'), "rb"))
    else:
        en2cn_dict = {}
    for i in range(len(input_tag_list)):
        if input_tag_list[i] in en2cn_dict.keys():
            list_trans.append(en2cn_dict[input_tag_list[i]])
        else:
            print("use youdao API......")
            trans_result = get_reuslt(translate(input_tag_list[i]))
            list_trans.append(trans_result)
            en2cn_dict[input_tag_list[i]] = trans_result
    pickle.dump(en2cn_dict, open(os.path.join(base_dir, 'en2cn_dict.txt'), "wb"))
 
    list_trans = list(set(list_trans))
    print("len of tags for translating is {}, and translate time(all) is {}".format(len(input_tag_list), time.clock()-translate_start))
    print("after translation tags are ")
    print(list_trans)

    synon_start = time.clock()
    if os.path.exists(os.path.join(base_dir, 'synonyms_words_dict.txt')):
        synonyms_words_dict = pickle.load(open(os.path.join(base_dir, 'synonyms_words_dict.txt'), "rb"))
    else:
        synonyms_words_dict = {}
    for i in range(len(list_trans)):
        if list_trans[i] in synonyms_words_dict.keys():
            if len(synonyms_words_dict[list_trans[i]])==0:
                continue
            final_input_tag_list.append(synonyms_words_dict[list_trans[i]])
        else:
            print("using synonyms_words package.......")
            synonyms_words = synonyms.nearby(list_trans[i])
            d=dict(zip(synonyms_words[0],synonyms_words[1]))
            synonyms_words = [k for k,v in d.items() if v >=0.7]
            if len(synonyms_words[:3])==0:
                synonyms_words_dict[list_trans[i]] = []
                continue
            final_input_tag_list.append(synonyms_words[:3])
            synonyms_words_dict[list_trans[i]] = synonyms_words[:3]
    pickle.dump(synonyms_words_dict, open(os.path.join(base_dir, 'synonyms_words_dict.txt'), "wb"))

    print("len of words to find synonyms_words is {}, and time(all) for synonyms_words is {}".format(len(list_trans), time.clock()-synon_start))
    print("tags and their synonyms_words are")
    print(final_input_tag_list)
    
    if random_tags:
        original_final_input_tag_list = final_input_tag_list.copy()
        final_input_tag_list = random.sample(final_input_tag_list, int(0.75*len(final_input_tag_list)))
        for i in range(len(final_input_tag_list)):
            original_final_input_tag_list.remove(final_input_tag_list[i])
        not_retrieved_tag_list = original_final_input_tag_list
        print("retrieved_tag_list is \n{}".format(final_input_tag_list))
        print("not_retrieved_tag_list is \n{}".format(not_retrieved_tag_list))

    retrieval_start = time.clock()
    retrieval_results = []
    for i in range(len(final_input_tag_list)):
        tag = final_input_tag_list[i]
        tmp_start = time.clock()
        tag_retrieval_result = retrieve_tag_baseon_dict(tag, dict_1, dict_2)
        retrieval_results+=tag_retrieval_result
    time_1 = time.clock()
    size = 0
    for i in range(len(final_input_tag_list)):
        size+=len(final_input_tag_list[i])
    print("the number of retrieved tags is {} and retrieval time is {}".format(size, time_1-retrieval_start))

    results = {}
    for i in retrieval_results:
        results[i] = results.get(i, 0) + 1 
    results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    output_results_index = [index[0] for index in results[:final_output_number+10]]

    max_match_number = results[0][1]
    if max_match_number==1 and random_tags: # 如果返回的匹配结果都只有1分, 就把没有检索的tag也检索一遍
        print('retrieval results based on the random tags are bad, I will retrieve the remain tags')
        second_retrieve_start = time.clock()
        for i in range(len(not_retrieved_tag_list)):
            tag = not_retrieved_tag_list[i]
            tmp_start = time.clock()
            tag_retrieval_result = retrieve_tag_baseon_dict(tag, dict_1, dict_2)
            retrieval_results+=tag_retrieval_result
        results = {}
        for i in retrieval_results:
            results[i] = results.get(i, 0) + 1 
        print("time used for retrieve the remain tags is {}".format(time.clock()-second_retrieve_start))
        results = sorted(results.items(), key=lambda item: item[1], reverse=True)
        output_results_index = [index[0] for index in results[:final_output_number+10]]
    new_results = list(set([new_all_lines[i][:-1] for i in output_results_index])) # 有的诗句出现了多次
    new_results = new_results[0:final_output_number]
    return new_results

def retrieve_tag_baseon_dict(tags, dict_1, dict_2):
    result_list = []
    for tag in tags:
        if len(tag)>=3:
            continue
        tmp_list = list(set(dict_1.get(tag, [])+dict_2.get(tag, [])))
        result_list = list(set(tmp_list+result_list))
    return result_list

if __name__ == "__main__":
    load_data_start = time.clock()
    shanglian_filename = os.path.join(base_dir, 'train.txt.up')
    dict_1_filename = os.path.join(base_dir, 'dict_1.txt')
    dicr_2_filename = os.path.join(base_dir, 'dict_2.txt')
    with open(shanglian_filename, 'r', encoding='utf-8') as in_file:
        shanglian_file_content = in_file.readlines()
    dict_1 = pickle.load(open(dict_1_filename, "rb"))
    dict_2 = pickle.load(open(dicr_2_filename, "rb"))
    load_data_end = time.clock()

    with open(os.path.join(base_dir, 'imgs/download.jpg'), 'rb') as f:
        call_api_start = time.clock()
        input_info = call_cv_api(f.read())

    start = time.clock()
    results = find_shanglian(input_info, shanglian_file_content, dict_1, dict_2, final_output_number=5) # tag_mode表示同义词是/否只进行一次匹配, 对应取值2/1

    print(results)# ['思潮如江河湖海水涌', '日落长河生丽水', '江河湖海四水归一', '江河湖海滔滔水', '江河湖海皆有水']
    print("load data takes {} seconds".format(load_data_end-load_data_start))
    print("find shanglian takes {} seconds".format(time.clock()-start))
    
