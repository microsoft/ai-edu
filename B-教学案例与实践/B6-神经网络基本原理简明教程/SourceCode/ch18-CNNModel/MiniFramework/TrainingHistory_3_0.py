# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import matplotlib.pyplot as plt
import pickle

from MiniFramework.EnumDef_6_0 import *

# 帮助类，用于记录损失函数值极其对应的权重/迭代次数
class TrainingHistory_3_0(object):
    def __init__(self, need_earlyStop = False, patience = 5):
        self.loss_train = []
        self.accuracy_train = []
        self.iteration_seq = []
        self.epoch_seq = []
        self.loss_val = []
        self.accuracy_val = []
        self.counter = 0
        self.max_vld_acc = 0
        # for early stop
        self.early_stop = need_earlyStop
        self.patience = patience
        self.patience_counter = 0
        self.last_vld_loss = float("inf")

    def Add(self, epoch, total_iteration, loss_train, accuracy_train, loss_vld, accuracy_vld, stopper):
        self.iteration_seq.append(total_iteration)
        self.epoch_seq.append(epoch)
        self.loss_train.append(loss_train)
        self.accuracy_train.append(accuracy_train)
        if loss_vld is not None:
            self.loss_val.append(loss_vld)
        if accuracy_vld is not None:
            self.accuracy_val.append(accuracy_vld)

        if stopper is not None:
            if stopper.stop_condition == StopCondition.StopDiff:
                if len(self.loss_val) > 1:
                    if abs(self.loss_val[-1] - self.loss_val[-2]) < stopper.stop_value:
                        self.counter = self.counter + 1
                        if self.counter > 3:
                            return True
                    else:
                        self.counter = 0
                #end if
            #end if
        #end if

        if self.early_stop:
            if loss_vld < self.last_vld_loss:
                self.patience_counter = 0
                self.last_vld_loss = loss_vld
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    return True     # need to stop
            # end if
        # end if

        return False

    def IsMaximum(self, acc_vld):
        if acc_vld is not None:
            if acc_vld > self.max_vld_acc:
                self.max_vld_acc = acc_vld
                return True
        return False

    # 图形显示损失函数值历史记录
    def ShowLossHistory(self, title, xcoord, xmin=None, xmax=None, ymin=None, ymax=None):
        fig = plt.figure(figsize=(12,5))

        axes = plt.subplot(1,2,1)
        if xcoord == XCoordinate.Iteration:
            p2, = axes.plot(self.iteration_seq, self.loss_train)
            p1, = axes.plot(self.iteration_seq, self.loss_val)
            axes.set_xlabel("iteration")
        elif xcoord == XCoordinate.Epoch:
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
        if xcoord == XCoordinate.Iteration:
            p2, = axes.plot(self.iteration_seq, self.accuracy_train)
            p1, = axes.plot(self.iteration_seq, self.accuracy_val)
            axes.set_xlabel("iteration")
        elif xcoord == XCoordinate.Epoch:
            p2, = axes.plot(self.epoch_seq, self.accuracy_train)
            p1, = axes.plot(self.epoch_seq, self.accuracy_val)
            axes.set_xlabel("epoch")
        #end if
        axes.legend([p1,p2], ["validation","train"])
        axes.set_title("Accuracy")
        axes.set_ylabel("accuracy")
        
        plt.suptitle(title)
        plt.show()
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