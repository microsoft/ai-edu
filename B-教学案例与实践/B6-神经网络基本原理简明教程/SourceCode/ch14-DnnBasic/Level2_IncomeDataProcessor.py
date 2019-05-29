# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from MiniFramework.NeuralNet import *
from MiniFramework.Optimizer import *
from MiniFramework.LossFunction import *
from MiniFramework.Parameters import *
from MiniFramework.WeightsBias import *
from MiniFramework.ActivatorLayer import *
from MiniFramework.DataReader import *

import numpy as np
import csv

data_file_name = "../../Data/PM25_data_20100101_20141231.csv"
train_data = "../../Data/PM25_Train.npz"
test_data = "../../Data/PM25_Test.npz"

class PM25DataProcessor(object):
    def PrepareData(self, csv_file):
        self.XTrainData = None
        self.XTestData = None
        self.YTrainData = None
        self.YTestData = None
        i = 0
        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            array = np.zeros((1,10))
            array_y = np.zeros((1,1))
            for row in reader:
                if row[5] == 'NA' or row[5] == 'pm2.5':
                    continue
                # don't need to read 'No' and 'Year'
                array[0,0] = row[2]
                array[0,1] = row[3]
                array[0,2] = row[4]
                array[0,3] = row[6]
                array[0,4] = row[7]
                array[0,5] = row[8]
                array[0,6] = self.convertWindDirectionToNumber(row[9])
                array[0,7] = row[10]
                array[0,8] = row[11]
                array[0,9] = row[12]
                array_y[0,0] = row[5]

                if int(row[1]) < 2014:
                    if self.XTrainData is None:
                        self.XTrainData = array
                    else:
                        self.XTrainData = np.vstack((self.XTrainData, array))
                    #end if
                    if self.YTrainData is None:
                        self.YTrainData = array_y[0,0]
                    else:
                        self.YTrainData = np.vstack((self.YTrainData, array_y))
                    #end if
                else:
                    if self.XTestData is None:
                        self.XTestData = array
                    else:
                        self.XTestData = np.vstack((self.XTestData, array))
                    #end if
                    if self.YTestData is None:
                        self.YTestData = array_y[0,0]
                    else:
                        self.YTestData = np.vstack((self.YTestData, array_y))
                    #end if
                #end if
                i = i+1
                if i % 100 == 0:
                    print(i)
            #end for
            print(self.XTrainData.shape, self.YTrainData.shape, self.XTestData.shape, self.YTestData.shape)
            np.savez(train_data, data=self.XTrainData, label=self.YTrainData)
            np.savez(test_data, data=self.XTestData, label=self.YTestData)

    def convertWindDirectionToNumber(self, wd):
        if wd == 'cv':
            return 0
        elif wd == 'N':
            return 1
        elif wd == 'E':
            return 2
        elif wd == 'S':
            return 4
        elif wd == 'W':
            return 8
        elif wd == 'NE':
            return 3
        elif wd == 'NW':
            return 9
        elif wd == 'SE':
            return 6
        elif wd == 'SW':
            return 12
        else:
            raise("error")


    def NormalizeX(self):
        pass

if __name__ == '__main__':
    dr = PM25DataProcessor()
    dr.PrepareData(data_file_name)
