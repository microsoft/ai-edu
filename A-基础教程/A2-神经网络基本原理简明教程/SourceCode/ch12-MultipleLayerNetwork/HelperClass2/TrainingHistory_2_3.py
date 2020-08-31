# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

"""
Version 2.3
what's new?
- add epoch/iteration as parameter in showLossHistory
"""

import matplotlib.pyplot as plt
import pickle

# 帮助类，用于记录损失函数值极其对应的权重/迭代次数
class TrainingHistory_2_3(object):
    def __init__(self):
        self.loss_train = []
        self.accuracy_train = []
        self.iteration_seq = []
        self.epoch_seq = []

        self.loss_val = []
        self.accuracy_val = []
       
    def Add(self, epoch, total_iteration, loss_train, accuracy_train, loss_vld, accuracy_vld):
        self.iteration_seq.append(total_iteration)
        self.epoch_seq.append(epoch)
        self.loss_train.append(loss_train)
        self.accuracy_train.append(accuracy_train)
        if loss_vld is not None:
            self.loss_val.append(loss_vld)
        if accuracy_vld is not None:
            self.accuracy_val.append(accuracy_vld)

        return False

    # 图形显示损失函数值历史记录
    def ShowLossHistory(self, params, x="epoch", xmin=None, xmax=None, ymin=None, ymax=None):
        fig = plt.figure(figsize=(12,5))

        axes = plt.subplot(1,2,1)
        if x == "iteration":
            p2, = axes.plot(self.iteration_seq, self.loss_train)
            p1, = axes.plot(self.iteration_seq, self.loss_val)
            axes.set_xlabel("iteration")
        elif x == "epoch":
            p2, = axes.plot(self.epoch_seq, self.loss_train)
            p1, = axes.plot(self.epoch_seq, self.loss_val)
            axes.set_xlabel("epoch")
        #end if
        axes.legend([p1,p2], ["validation","train"])
        axes.set_title("Loss")
        axes.set_ylabel("loss")
        if xmin != None or xmax != None or ymin != None or ymax != None:
            axes.axis([xmin, xmax, ymin, ymax])
        
        axes = plt.subplot(1,2,2)
        if x == "iteration":
            p2, = axes.plot(self.iteration_seq, self.accuracy_train)
            p1, = axes.plot(self.iteration_seq, self.accuracy_val)
            axes.set_xlabel("iteration")
        elif x == "epoch":
            p2, = axes.plot(self.epoch_seq, self.accuracy_train)
            p1, = axes.plot(self.epoch_seq, self.accuracy_val)
            axes.set_xlabel("epoch")
        #end if
        axes.legend([p1,p2], ["validation","train"])
        axes.set_title("Accuracy")
        axes.set_ylabel("accuracy")
        
        title = params.toString()
        plt.suptitle(title)
        plt.show()
        return title

    def ShowLossHistory4(self, axes, params, xmin=None, xmax=None, ymin=None, ymax=None):
        p2, = axes.plot(self.epoch_seq, self.loss_train)
        p1, = axes.plot(self.epoch_seq, self.loss_val)
        title = params.toString()
        axes.set_title(title)
        axes.set_xlabel("epoch")
        axes.set_ylabel("loss")
        if xmin != None and ymin != None:
            axes.axis([xmin, xmax, ymin, ymax])
        return title

    def GetEpochNumber(self):
        return self.epoch_seq[-1]

    def GetLatestAverageLoss(self, count=10):
        total = len(self.loss_val)
        if count >= total:
            count = total
        tmp = self.loss_val[total-count:total]
        return sum(tmp)/count

    def Dump(self, file_name):
        f = open(file_name, 'wb')
        pickle.dump(self, f)

    def Load(file_name):
        f = open(file_name, 'rb')
        lh = pickle.load(f)
        return lh

# end class