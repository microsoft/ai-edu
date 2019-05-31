
import numpy as np
import os
import csv

data_folder = "E:\\Backup\\fonts\\"
train_data = "../../Data/Font_Train.npz"

def ReadOneFile(name):
    X = None
    Y = None

    with open(name, newline='') as f:
        reader = csv.reader(f)
        array_x = np.zeros((1,400))
        array_y = np.zeros((1,1))
        i=0
        for row in reader:
            if i==0:
                i=1
                continue
            aaa = (int)(row[2])
            if (aaa >= 65 and aaa <= 90) or (aaa >= 97 and aaa <= 122):  # AZ,az
                array_y[0,0] = aaa
                for i in range(400):
                    array_x[0,i] = (int)(row[12+i])
                #end for
                if X is None:
                    X = array_x
                    Y = array_y
                else:
                    X = np.vstack((X, array_x))
                    Y = np.vstack((Y, array_y))
                #end if
            #end if
    return X,Y

def ReadFiles(dir):
    X = None
    Y = None
    for f in os.listdir(dir):
        file_name = os.path.join(dir,f)
        if os.path.isfile(file_name):
            print(file_name)
            x,y = ReadOneFile(file_name)
            if X is None:
                X = x
                Y = y
            else:
                if x is not None and y is not None:
                    X = np.vstack((X, x))
                    Y = np.vstack((Y, y))
            #end if
        #end if
    #end for
    np.savez(train_data, data=X, label=Y)

if __name__ == '__main__':
    ReadFiles(data_folder)