# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import string
import random
from MiniFramework.DataReader_2_0 import *
from MiniFramework.EnumDef_6_0 import *

all_letters = string.ascii_letters[0:26] + " ,'"
num_letter = len(all_letters)
max_letter = 20

class NameData(object):
    def __init__(self, name, lang, lang_id):
        self.name = name
        self.language = lang
        self.lang_id = lang_id
        self.num_step = len(self.name)

    def ToOneHot(self, num_category):
        name_onehot = np.zeros((self.num_step, num_letter))
        lang_onehot = np.zeros((1, num_category))
        for i in range(self.num_step):
            id = all_letters.find(self.name[i])
            name_onehot[i,id] = 1
        #endfor
        lang_onehot[0,self.lang_id] = 1
        return name_onehot, lang_onehot

class NameDataReader(DataReader_2_0):
    def __init__(self):
        self.name_data = []
        self.language_list = []
        self.num_feature = num_letter
        self.max_step = max_letter
        for i in range(max_letter):
            self.name_data.append([])

        self.batch_id = 0
        self.name_id = 0

    def ReadData(self, filename):
        file = open(filename, 'r')
        lines = file.readlines()
        self.num_train = len(lines)
        for line in lines:
            tmp = line.split("\t")
            name = tmp[0].strip()
            language = tmp[1].strip()
            len_name = len(name)
            lang_id = self.getLanguageId(language)
            nd = NameData(name, language, lang_id)
            self.name_data[len_name].append(nd)
        #endfor
        file.close()
        self.num_category = len(self.language_list)

        self.X = []
        self.Y = []
        num_Y = 0
        for i in range(max_letter):
            num_step = i  # how many letters in the name, aka. timestep
            num_names = len(self.name_data[i])  # how many names in this list
            if (num_names > 0):
                Xi = np.zeros((num_names, num_step, num_letter))            
                Yi = np.zeros((num_names, self.num_category))
                self.X.append(Xi)
                self.Y.append(Yi)
                for j in range(num_names):
                    nd = self.name_data[i][j]
                    Xi[j], Yi[j] = nd.ToOneHot(self.num_category)
                #end for
            #end if
        #end for       

    def getLanguageId(self, language):
        try:
            lang_id = self.language_list.index(language)
        except:
            self.language_list.append(language)
            lang_id = self.language_list.index(language)
        finally:
            return lang_id

    def ResetPointer(self):
        self.batch_id = 0
        self.name_id = 0

    def GetBatchTrainSamples(self, batch_size):
        if (self.batch_id >= len(self.X)):
            self.batch_id = 0
            return None, None

        start = self.name_id
        end = start + batch_size
        x = self.X[self.batch_id][start:end]
        y = self.Y[self.batch_id][start:end]
        self.name_id = end
        if (self.name_id >= len(self.X[self.batch_id])):
            self.name_id = 0
            self.batch_id += 1
        return x, y
    
    def GetRandomBatchTrainSamples(self, batch_size):
        idx1 = random.randint(0, len(self.X)-1)
        idx2 = random.randint(0, max(self.X[idx1].shape[0]-batch_size,0))
        x = self.X[idx1][idx2:idx2+batch_size]
        y = self.Y[idx1][idx2:idx2+batch_size]
        return x,y

    def GenerateValidationSet(self, k):
        self.dev_x = []
        self.dev_y = []
        for i in range(k):
            x,y = self.GetRandomBatchTrainSamples(1)
            self.dev_x.append(x)
            self.dev_y.append(y)
        self.num_dev = k

    def GetValidationSet(self):
        return self.dev_x, self.dev_y

    def Shuffle(self):
        count = len(self.X)
        self.shuffle_list()
        for i in range(count):
            self.X[i], self.Y[i] = self.shuffle_array(self.X[i], self.Y[i])
        # end for

    def GetLanguageById(self, lang_id):
        return self.language_list[lang_id]

    def GetLanguageIdByName(self, name):
        return self.language_list.index(name)

    def shuffle_list(self):
        seed = random.randint(0,100)
        random.seed(seed)
        random.shuffle(self.X)
        random.seed(seed)
        random.shuffle(self.Y)

    def shuffle_array(self, x, y):
        seed = np.random.randint(0,100)
        np.random.seed(seed)
        x_new = np.random.permutation(x)
        np.random.seed(seed)
        y_new = np.random.permutation(y)
        return x_new, y_new
        