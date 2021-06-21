# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import argparse
from collections import OrderedDict
from sklearn import model_selection

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helper.file import load_pickle, save_pickle, save_json


def main(input_path: str, output_path: str, random_state_file: str):
    statistics = OrderedDict()
    origin_data = pd.read_csv(os.path.join(input_path, 'data.csv'))
    statistics['originDatasetSize'] = origin_data.shape[0]
    origin_data.replace(np.nan, '', inplace=True)
    data = origin_data.replace('', np.nan)
    data = data.dropna()
    statistics['withDocDatasetSize'] = data.shape[0]
    os.makedirs(output_path, exist_ok=True)
    if random_state_file:
        random_state = load_pickle(random_state_file)
        np.random.set_state(random_state)
    save_pickle(np.random.get_state(),
                os.path.join(output_path, 'random_state.pkl'))
    train, test = model_selection.train_test_split(data, test_size=0.2)
    valid, test = model_selection.train_test_split(test, test_size=0.5)
    statistics['trainDatasetSize'] = train.shape[0]
    statistics['validDatasetSize'] = valid.shape[0]
    statistics['testDatasetSize'] = test.shape[0]
    train.to_csv(os.path.join(output_path, 'train.csv'), index=False)
    valid.to_csv(os.path.join(output_path, 'valid.csv'), index=False)
    test.to_csv(os.path.join(output_path, 'test.csv'), index=False)
    data.to_csv(os.path.join(output_path, 'withdoc.csv'), index=False)
    origin_data.to_csv(os.path.join(output_path, 'total.csv'), index=False)
    for k, v in statistics.items():
        print('%s: %d' % (k, v))
    save_json(statistics, os.path.join(output_path, 'statistics.json'))


if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',
                        help='input path containing cleaned csv file')
    parser.add_argument('output_path',
                        help='output path that split data is written to')
    parser.add_argument('--random_state', help='path to the random state file')
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.random_state)
