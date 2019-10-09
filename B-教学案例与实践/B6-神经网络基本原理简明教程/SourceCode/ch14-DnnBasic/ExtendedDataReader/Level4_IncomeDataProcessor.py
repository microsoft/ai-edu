# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import csv

train_data = "../../Data/adult.data"
test_data = "../../Data/adult.test"

train_data_npz = "../../Data/ch14.Income.train.npz"
test_data_npz = "../../Data/ch14.Income.test.npz"

workclass_list = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
education_list = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
marital_list = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
occupation_list = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
relationship_list = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
race_list = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
sex_list = ["Female", "Male"]
native_country_list = ["United-States","Cambodia","England","Puerto-Rico","Canada","Germany","Outlying-US(Guam-USVI-etc)","India","Japan","Greece","South","China","Cuba","Iran","Honduras","Philippines","Italy","Poland","Jamaica","Vietnam","Mexico","Portugal","Ireland","France","Dominican-Republic","Laos","Ecuador","Taiwan","Haiti","Columbia","Hungary","Guatemala","Nicaragua","Scotland","Thailand","Yugoslavia","El-Salvador","Trinadad&Tobago","Peru","Hong","Holand-Netherlands"]
label = ["<=50K", ">50K"]

class IncomeDataProcessor(object):
    def PrepareData(self, csv_file, data_npz):
        self.XData = None
        self.YData = None
        i = 0
        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            array_x = np.zeros((1,14))
            array_y = np.zeros((1,1))
            for row in reader:
                if len(row) == 0:
                    continue
                try:
                    row.index(" ?")
                    print("found ?, discard data")
                    continue
                except:
                    array_x[0,0] = row[0]
                    array_x[0,1] = workclass_list.index(row[1].strip())
                    array_x[0,2] = row[2]
                    array_x[0,3] = education_list.index(row[3].strip())
                    array_x[0,4] = row[4]
                    array_x[0,5] = marital_list.index(row[5].strip())
                    array_x[0,6] = occupation_list.index(row[6].strip())
                    array_x[0,7] = relationship_list.index(row[7].strip())
                    array_x[0,8] = race_list.index(row[8].strip())
                    array_x[0,9] = sex_list.index(row[9].strip())
                    array_x[0,10] = row[10]
                    array_x[0,11] = row[11]
                    array_x[0,12] = row[12]
                    array_x[0,13] = native_country_list.index(row[13].strip())
                    array_y[0,0] = label.index(row[14].strip())

                    if self.XData is None:
                        self.XData = array_x
                    else:
                        self.XData = np.vstack((self.XData, array_x))
                    #end if
                    if self.YData is None:
                        self.YData = array_y[0,0]
                    else:
                        self.YData = np.vstack((self.YData, array_y))
                    #end if
   
                i = i+1
                if i % 100 == 0:
                    print(i)
            #end for
            np.savez(data_npz, data=self.XData, label=self.YData)

if __name__ == '__main__':
    dr = IncomeDataProcessor()
    dr.PrepareData(train_data, train_data_npz)
    dr.PrepareData(test_data, test_data_npz)
