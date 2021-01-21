# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import os
import sys
import numpy as np

class CoupletReader(object):
    def __init__(self, train_dir, test_dir):
        self.train_dir = os.path.abspath(train_dir)
        self.test_dir = os.path.abspath(test_dir)
        self.train_up = self.ReadTxt(os.path.join(self.train_dir, "in.txt"))
        self.train_down = self.ReadTxt(os.path.join(self.train_dir, "out.txt"))
        self.test_up = self.ReadTxt(os.path.join(self.test_dir, "in.txt"))
        self.test_down = self.ReadTxt(os.path.join(self.test_dir, "out.txt"))

    def ReadTxt(self, filename, encoding='utf-8'):
        lines=[]
        with open(filename, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                lines.append(line)
        return lines


class PoetryReGen(object):
    def __init__(self, poetry_file):
        self.poetry_file = poetry_file

    def ReadData(self):
        content = np.load(self.poetry_file)
        self.data = content['data']
        self.ix2word = content['ix2word'].item()
        self.word2ix = content['word2ix'].item()
        self.vocabsize = len(self.word2ix)

    def DataFilter(self):
        filter_5 = []
        filter_7 = []
        filter_others = []
        print("Start reading data ...")
        for i in range(self.data.shape[0]):
            line = []
            for j in range(self.data.shape[1]):
                idx = self.data[i][j]
                w = self.ix2word[idx]
                if (w in {u'</s>', u'<START>', u'<EOP>'}):
                    continue
                line.append(idx)
                if (w == u'。'):
                    if (len(line) is 12):
                        filter_5.append(line)
                    elif (len(line) is 16):
                        filter_7.append(line)
                    else:
                        filter_others.append(line)
                    line = []
        filter_5 = np.asarray(filter_5)
        filter_7 = np.asarray(filter_7)
        filter_others = np.asarray(filter_others)

        print("shape of filter_5 file: ", filter_5.shape)
        print("shape of filter_7 file: ", filter_7.shape)
        print("shape of filter_others file: ", filter_others.shape)

        print("start writing to file ...")
        np.savez_compressed("filter_5.npz", data=filter_5)
        np.savez_compressed("filter_7.npz", data=filter_7)
        np.savez_compressed("filter_others.npz", data=filter_others)
        np.savez_compressed("index.npz", ix2word=self.ix2word, word2ix=self.word2ix)
        print("Done.")


    def DataAnalysis(self):
        print("data shape: ", self.data.shape, " vocab size: ", self.vocabsize)
        print("data[2360]:")
        line = []
        for i in self.data[2360]:
            line.append(self.ix2word[i])
        print(line)
        # print("vocabs: ")
        # for k, v in self.ix2word.items():
        #     print(k, " ", v)



if __name__=='__main__':
    # test couplet data
    # train_dir = "coupletdata/train"
    # test_dir = "coupletdata/test"
    # data = CoupletData(train_dir, test_dir)
    # print(data.train_up)

    # test tang data
    tang_data = "tang.npz"
    pr = PoetryReGen(tang_data)
    pr.ReadData()
    pr.DataFilter()
    # pr.DataAnalysis()