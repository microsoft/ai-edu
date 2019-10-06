
from io import open
import glob
import os
import unicodedata
import string
import numpy as np

all_letters = string.ascii_letters + " .,;'"
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

def generate_array():
    files = findFiles('../../data/names/*.txt')
    category = len(files)
    
    index = 0
    X = None
    Y = None
    def_len = 20
    max_len = 0
    for filename in findFiles('../../data/names/*.txt'):
        language = os.path.splitext(os.path.basename(filename))[0]
        language_index.append(language)
        names = readLines(filename)
        x = np.zeros((len(names),def_len,len(lower_letters)), dtype=np.int16)
        y = np.zeros((len(names),category), dtype=np.int16)
        for i in range(len(names)):
            len_name = len(names[i])
            if (len_name > max_len):
                max_len = len_name
                print(names[i])
            for j in range(len_name):
                letter_id = lower_letters.find(names[i][j])
                x[i,j,letter_id] = 1
            #endfor
            language_id = language_index.index(language)
            y[i,language_id] = 1
        #endfor
        if (X is None):
            X = x
            Y = y
        else:
            X = np.concatenate((X,x))
            Y = np.concatenate((Y,y))

    print(max_len)
    X1 =np.delete(X, max_len - def_len, axis=1)
    print(X.shape)
    print(X1.shape)

def generate_file():
    files = findFiles('../../data/names/*.txt')
    category = len(files)
    file = open("../../data/name_language.txt", mode='w')

    index = 0
    X = None
    Y = None
    def_len = 20
    max_len = 0
    for filename in findFiles('../../data/names/*.txt'):
        language = os.path.splitext(os.path.basename(filename))[0]
        language_index.append(language)
        names = readLines(filename)
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
