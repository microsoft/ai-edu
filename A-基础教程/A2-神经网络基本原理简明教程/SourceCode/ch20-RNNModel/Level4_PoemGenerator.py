# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
from pathlib import Path
from Level2_LSTM import *

poetry5_file = "filter_5.npz"
poetry7_file = "filter_7.npz"
index_file = "index.npz"

class ReadData(object):
    def __init__(self, index_path):
        indexpath = Path(index_path)
        if indexpath.exists():
            index = np.load(index_path)
            self.word2ix = index['word2ix']
            self.ix2word = index['ix2word']
            self.vocab_size = len(self.word2ix)
        else:
            raise Exception("Cannot find index file!!!")

    def LoadPoetry(self, poetry_path):
        poetrypath = Path(poetry_path)
        if poetrypath.exists():
            self.data = np.load(poetry_path)['data']
        else:
            raise Exception("Cannot find poetry file!!!")

        ## Split data to training, validation and test sets


    def (self):
        voc = ["UNK"]
        voc.extend(self.ReadData(self.vocfile))
        self.vocab = voc
        self.vocabsize = len(self.vocab)

    def BuildVector(self):
        veclist = []
        size = self.vocabsize
        for i in range(size):
            vec = [0 for _ in range(size)]
            vec[i] = 1
            veclist.append(vec)
        self.vec = veclist

    def Read(self):
        self.poetry = self.ReadData(self.pfile)
        self.ReadVocab()
        self.BuildVector()
        return (self.poetry, self.vocab, self.vec)


def load_data():
    p = ReadPoem(poetry, vocab)
    data = p.Read()
    return data


class net(object):



if __name__ == '__main__':