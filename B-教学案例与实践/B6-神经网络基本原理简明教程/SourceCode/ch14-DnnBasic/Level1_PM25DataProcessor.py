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
            array_x = np.zeros((1,10))
            array_y = np.zeros((1,1))
            for row in reader:
                if row[5] == 'NA' or row[5] == 'pm2.5':
                    continue
                # don't need to read 'No' and 'Year'
                array_x[0,0] = row[2]   # month
                array_x[0,1] = row[3]   # day
                array_x[0,2] = row[4]   # hour
                array_x[0,3] = row[6]   # DEWP
                array_x[0,4] = row[7]   # TEMP
                array_x[0,5] = row[8]   # PRES
                array_x[0,6] = self.convertWindDirectionToNumber(row[9])
                array_x[0,7] = row[10]  # WindSpeed
                array_x[0,8] = row[11]  # snow
                array_x[0,9] = row[12]  # rain
                array_y[0,0] = self.convertPM25ValueToGrade((int)(row[5]))   # label

                day = int(row[3])
                if day % 7 != 0:
                    if self.XTrainData is None:
                        self.XTrainData = array_x
                    else:
                        self.XTrainData = np.vstack((self.XTrainData, array_x))
                    #end if
                    if self.YTrainData is None:
                        self.YTrainData = array_y[0,0]
                    else:
                        self.YTrainData = np.vstack((self.YTrainData, array_y))
                    #end if
                else:
                    if self.XTestData is None:
                        self.XTestData = array_x
                    else:
                        self.XTestData = np.vstack((self.XTestData, array_x))
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

    def convertPM25ValueToGrade(self, value):
        if value <= 35:
            return 0
        elif value <= 75:
            return 1
        elif value <= 115:
            return 2
        elif value <= 150:
            return 3
        elif value <= 250:
            return 4
        else:
            return 5

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
