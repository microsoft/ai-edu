# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

# 帮助类，用于记录损失函数值极其对应的权重/迭代次数
class TrainingTrace(object):
    def __init__(self, need_earlyStop = False, patience = 5):
        self.loss_train = []
        self.accuracy_train = []
        self.iteration_seq = []
        self.epoch_seq = []

        self.loss_val = []
        self.accuracy_val = []
       
        # for early stop
        self.min_loss_index = -1
        self.early_stop = need_earlyStop
        self.patience = patience
        self.patience_counter = 0
        self.last_vld_loss = float("inf")

    def Add(self, epoch, total_iteration, loss_train, accuracy_train, loss_vld, accuracy_vld):
        self.iteration_seq.append(total_iteration)
        self.epoch_seq.append(epoch)
        self.loss_train.append(loss_train)
        self.accuracy_train.append(accuracy_train)
        if loss_vld is not None:
            self.loss_val.append(loss_vld)
        if accuracy_vld is not None:
            self.accuracy_val.append(accuracy_vld)

        if self.early_stop:
            if loss_vld < self.last_vld_loss:
                self.patience_counter = 0
                self.last_vld_loss = loss_vld
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    return True     # need to stop
            # end if
        return False

    # 图形显示损失函数值历史记录
    def ShowLossHistory(self, params, xmin=None, xmax=None, ymin=None, ymax=None):
        fig = plt.figure(figsize=(12,5))

        axes = plt.subplot(1,2,1)
        #p2, = axes.plot(self.iteration_seq, self.loss_train)
        #p1, = axes.plot(self.iteration_seq, self.loss_val)
        p2, = axes.plot(self.epoch_seq, self.loss_train)
        p1, = axes.plot(self.epoch_seq, self.loss_val)
        axes.legend([p1,p2], ["validation","train"])
        axes.set_title("Loss")
        axes.set_ylabel("loss")
        axes.set_xlabel("epoch")
        if xmin != None or xmax != None or ymin != None or ymax != None:
            axes.axis([xmin, xmax, ymin, ymax])
        
        axes = plt.subplot(1,2,2)
        #p2, = axes.plot(self.iteration_seq, self.accuracy_train)
        #p1, = axes.plot(self.iteration_seq, self.accuracy_val)
        p2, = axes.plot(self.epoch_seq, self.accuracy_train)
        p1, = axes.plot(self.epoch_seq, self.accuracy_val)
        axes.legend([p1,p2], ["validation","train"])
        axes.set_title("Accuracy")
        axes.set_ylabel("accuracy")
        axes.set_xlabel("epoch")
        
        title = params.toString()
        plt.suptitle(title)
        plt.show()
        return title



# end class