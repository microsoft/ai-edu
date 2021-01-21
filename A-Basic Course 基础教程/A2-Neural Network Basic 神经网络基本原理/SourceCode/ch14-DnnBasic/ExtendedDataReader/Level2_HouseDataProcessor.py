# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import csv
import copy

data_file_name = "../../Data/kc_house_data.csv"     # download from Kc
train_data_file_name = "../../Data/kc_train.csv"    # download from DC
test_data_file_name = "../../Data/kc_test.csv"      # download from DC
train_file = "../../Data/ch14.house.train.npz"
test_file = "../../Data/ch14.house.test.npz"

class HouseDataProcessor(object):
    def PrepareData(self, csv_file, npz_file):
        self.X = None
        self.Y = None
        i = 0
        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            array_x = np.zeros((1,12))
            array_y = np.zeros((1,1))
            for row in reader:
                
                array_x[0,0] = row[2]   # bedrooms
                array_x[0,1] = row[3]   # bathrooms
                array_x[0,2] = row[4]   # sqft living
                array_x[0,3] = row[5]   # sqft lot
                array_x[0,4] = row[6]   # floors
                array_x[0,5] = row[7]  # grade
                array_x[0,6] = row[8]  # sqft above
                array_x[0,7] = row[9]  # sqft base
                array_x[0,8] = row[10]  # year built
                if (int)(row[11]) != 0:
                    array_x[0,8] = row[11]  # year renovate
                array_x[0,9] = row[12]  # latitude
                array_x[0,10] = row[13]  # longitude
                array_x[0,11] = row[0]  # date
      

                array_y[0,0] = (float)(row[1])   # label

                if self.X is None:
                    self.X = array_x.copy()
                else:
                    self.X = np.vstack((self.X, array_x))
                #end if
                if self.Y is None:
                    self.Y = array_y[0,0]
                else:
                    self.Y = np.vstack((self.Y, array_y))
                #end if

                i = i+1
                if i % 100 == 0:
                    print(i)
            #end for
        #end with

        np.savez(npz_file, data=self.X, label=self.Y)

class HouseSingleDataProcessor(object):
    def PrepareData(self, csv_file):
        self.X = None
        self.Y = None
        i = 0
        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            array_x = np.zeros((1,13))
            array_y = np.zeros((1,1))
            for row in reader:
                if i == 0:
                    i += 1
                    continue
                # don't need to read 'No' and 'Year'
                array_x[0,0] = row[3]   # bedrooms
                array_x[0,1] = row[4]   # bathrooms
                array_x[0,2] = row[5]   # sqft living
                array_x[0,3] = row[6]   # sqft lot
                array_x[0,4] = row[7]   # floors
                array_x[0,5] = row[8]   # waterfront
                array_x[0,6] = row[10]   # condition
                array_x[0,7] = row[11]  # grade
                array_x[0,8] = row[12]  # sqft above
                array_x[0,9] = row[13]  # sqft base
                array_x[0,10] = row[14]  # year built
                if (int)(row[15]) != 0:
                    array_x[0,10] = (int)(row[15])
                #array_x[0,11] = row[15]  # year renovate
                array_x[0,11] = row[17]  # latitude
                array_x[0,12] = row[18]  # longitude
                #array_x[0,13] = row[0]  # sale date


                array_y[0,0] = (float)(row[2])   # label

                if self.X is None:
                    self.X = array_x.copy()
                else:
                    self.X = np.vstack((self.X, array_x))
                #end if
                if self.Y is None:
                    self.Y = array_y[0,0]
                else:
                    self.Y = np.vstack((self.Y, array_y))
                #end if

                i = i+1
                if i % 100 == 0:
                    print(i)
            #end for
        #end with

        self.RandomSplit(self.X, self.Y, 0.2)
 

    def RandomSplit(self, x, y, ratio=0.2):
        assert(x.shape[0] == y.shape[0])
        total = x.shape[0]
        seed = np.random.randint(0,100)
        np.random.seed(seed)
        np.random.shuffle(x)
        np.random.seed(seed)
        np.random.shuffle(y)

        testcount = (int)(total * ratio)
        
        x_test = x[0:testcount,:]
        x_train = x[testcount:,:]
        y_test = y[0:testcount,:]
        y_train = y[testcount:,:]

        np.savez(train_file, data=x_train, label=y_train)
        np.savez(test_file, data=x_test, label=y_test)
        
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

if __name__ == '__main__':
    #dr = HouseDataProcessor()
    #dr.PrepareData(train_data_file_name, train_data)
    #dr.PrepareData(test_data_file_name, test_data)
    dr = HouseSingleDataProcessor();
    dr.PrepareData(data_file_name)