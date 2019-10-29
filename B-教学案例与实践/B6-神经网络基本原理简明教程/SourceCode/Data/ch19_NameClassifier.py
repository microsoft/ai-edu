# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from io import open
import glob
import os
import unicodedata
import string
import numpy as np

#all_letters = string.ascii_letters + " .,;'"
all_letters = string.ascii_letters
lower_letters = string.ascii_letters[0:26] + " .,;'"
language_index = []

def findFiles(path): 
    return glob.glob(path)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    ).lower()

#print(unicodeToAscii('Ślusàrski'))

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

def dedup(names):
    dict = {}
    for name in names:
        if (name in dict):
            continue
        else:
            dict[name] = 1
    return list(dict.keys())

def generate_file():
    files = findFiles('../../data/names/*.txt')
    category = len(files)
    file = open("../../data/ch19.name_language.txt", mode='w')

    index = 0
    X = None
    Y = None
    def_len = 20
    max_len = 0
    for filename in findFiles('../../data/names/*.txt'):
        language = os.path.splitext(os.path.basename(filename))[0]
        language_index.append(language)
        names = readLines(filename)
        names = dedup(names)
        x = np.zeros((len(names),1))
        language_id = language_index.index(language)
        for i in range(len(names)):
            name = names[i]
            line = str.format("{0}\t{1}\n", name, language)
            file.write(line)
        #endfor
    file.close()

if __name__=='__main__':
    generate_file()
