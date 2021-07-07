# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import argparse
from collections import OrderedDict

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helper.convert import save_codenn_dataset


def main(input_path, output_path, max_vocab_size, min_frequency):
    tasks = OrderedDict()
    for task_name in ['train', 'valid', 'test', 'withdoc', 'total']:
        task = pd.read_csv(os.path.join(input_path, '%s.csv' % task_name))
        task.replace(np.nan, '', inplace=True)
        tasks[task_name] = task
    os.makedirs(output_path, exist_ok=True)
    statistics = save_codenn_dataset(tasks, max_vocab_size,
                                     min_frequency, output_path)
    for k, v in statistics.items():
        print('%s: %d' % (k ,v))
    tasks['total'][['file', 'start', 'code', 'url']] \
        .to_csv(os.path.join(output_path, 'use.codemap.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',
                        help='input path that contains cleaned csvs')
    parser.add_argument('output_path',
                        help='output path that dataset should be generated into')
    parser.add_argument('--max_vocab_size', type=int, default=10000,
                        help='max vocabulary size')
    parser.add_argument('--min_frequency', type=int, default=2,
                        help='min appeared time for vocabulary')
    args = parser.parse_args()
    main(args.input_path, args.output_path,
         args.max_vocab_size, args.min_frequency)
