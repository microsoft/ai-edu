# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import pandas as pd
from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot

# load data
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')

def set_pollution_value(ds, start, end, start_value, end_value):
    step_value = (end_value - start_value) / (end - start)
    value = start_value
    for i in range(start, end, 1):
        ds.iat[i,3] = value
        value = value + step_value
    return ds

def get_previous_pollution_value(ds, row, col):
    prev_value = 0
    if (row - 1 >= 0):
        prev_value = ds.iat[row-1,col]
    return prev_value

def get_next_pollution_value(ds, row, col):
    next_value = 0
    if (row < len(ds)):
        next_value = ds.iat[row,col]
    return next_value

def get_pollution_class(value):
    if (value < 50):
        return 0
    elif (value < 100):
        return 1
    elif (value < 150):
        return 2
    elif (value < 200):
        return 3
    elif (value < 300):
        return 4
    else:
        return 5

#dataset = read_csv('../../data/PM25_data.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset = read_csv('../../data/PM25_data.csv')
dataset.drop('No', axis=1, inplace=True)
dataset.drop('year', axis=1, inplace=True)
dataset.drop('Ir', axis=1, inplace=True)
dataset.drop('Is', axis=1, inplace=True)

# manually specify column names
#dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.columns = ['month', 'day', 'hour', 'pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd']
#dataset.index.name = 'date'
dataset = dataset.replace('NW',9)
dataset = dataset.replace('NE',3)
dataset = dataset.replace('SE',6)
dataset = dataset.replace('SW',12)
ds = dataset.replace('cv',0)

total = len(ds)
# fill NaN pollution value
for i in range(total):
    if (pd.isnull(ds.iat[i,3])):
        start = i
        # read previous pollution value
        prev_value = get_previous_pollution_value(ds, i, 3)
        for j in range(i+1, total, 1):
            if (pd.isnull(ds.iat[j,3]) == False):
                end = j 
                break
        next_value = get_next_pollution_value(ds, j, 3)
        print(start,end)
        ds = set_pollution_value(ds, start, end, prev_value, next_value)
    
# process accumulated wind_speed value
prev_value = 0
prev_dir = 0
for i in range(total):
    current_dir = ds.iat[i,7]
    accumulated_value = ds.iat[i,8]
    if (current_dir == prev_dir):
        current_value = accumulated_value - prev_value
        ds.iat[i,8] = current_value
    #endif
    prev_value = accumulated_value
    prev_dir = current_dir

print(ds.head(24))

pollution_y = np.zeros((total,2))
pollution_y[:,0] = ds['pollution'].to_numpy()
for i in range(total):
    pollution_y[i,1] = get_pollution_class(pollution_y[i,0])

dataset.drop('pollution', axis=1, inplace=True)
pollution_x = ds.to_numpy()

np.savez("../../data/ch19_pm25_train.npz", data=pollution_x[0:total-8760], label=pollution_y[0:total-8760])
np.savez("../../data/ch19_pm25_test.npz", data=pollution_x[total-8760:], label=pollution_y[total-8760:])
