# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.
from Level2_LSTM import *

poetry = "Quatrain5.txt"
vocab = "vocab.txt"

class ReadPoem(object):
    def __init__(self, poetryfile, vocabfile):
        self.vocabsize = 0
        self.pfile = poetryfile
        self.vocfile = vocabfile

    def ReadData(self, filepath, encoding='utf-8'):
        with open(filepath, 'r', encoding=encoding) as f:
            flist = f.readlines()
        return flist

    def ReadVocab(self):
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