# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
import itertools
import argparse
from collections import OrderedDict

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helper.file import save_json
from helper.tokenize import split_name, get_language, \
    tokenize_docstring_for_codenn
from helper.extractor import unescape

tqdm.pandas(desc='Clean')


def main(input_path, output_path):
    statistics = OrderedDict()
    fd = open(os.path.join(input_path, 'data.csv'), encoding='utf-8',errors='replace')
    data = pd.read_csv(fd)
    statistics['originDatasetSize'] = data.shape[0]
    # Drop row contains empty column except `desc` and drop column `imported`
    data.replace('', np.nan, inplace=True)
    data.desc.replace(np.nan, '', inplace=True)
    data.drop(['imported'], axis=1, inplace=True)
    data = data.dropna()
    statistics['datasetSizeAfterDropNanRound1'] = data.shape[0]
    # Drop duplicates
    data.drop_duplicates(subset=['name','apis','tokens','desc'], inplace=True)
    statistics['datasetSizeAfterDropDuplicates'] = data.shape[0]
    # Process name column: remove short names and names which begin with `_`
    #   and split name
    data.name = data.name.str.replace('$', '_')
    data = data[~data.name.str.startswith('_') & (data.name.str.len() > 2)]
    statistics['datasetSizeAfterDropInternalAndShortName'] = data.shape[0]
    data.name = data.name.apply(lambda x: '|'.join(
        [i.lower() for i in split_name(x) if not i.isdigit()]))
    # Process apis column: use only last identifier of callee
    data.apis = data.apis.apply(lambda x: '|'.join(
        [i.split('.')[-1] for i in x.split('|')]))
    # Process tokens column: split every token, use only non-number lexeme, and
    #   turn them into lower form
    data.tokens = data.tokens.str.replace('$', '_')
    data.tokens = data.tokens.apply(lambda x: '|'.join(
        list(set(itertools.chain(
            *[[i.lower() for i in split_name(y) if not i.isdigit()]
              for y in x.split('|')])))))
    # Process desc column: pick first paragraph, tokenize
    data.desc = data.desc.apply(lambda x: re.sub(
        r'([* \n])+', ' ', next(filter(None, unescape(x).split('\n\n')), ''))
                                .strip())
    nlp = get_language()

    def tokenize_desc(x: str) -> str:
        try:
            return '|'.join(tokenize_docstring_for_codenn(x, nlp))
        except ValueError:
            return ''
    data.desc = data.desc.progress_apply(tokenize_desc)
    data.replace('', np.nan, inplace=True)
    data.desc.replace(np.nan, '', inplace=True)
    data = data.dropna()
    statistics['datasetSizeAfterDropNanRound2'] = data.shape[0]
    os.makedirs(output_path, exist_ok=True)
    data.to_csv(os.path.join(output_path, 'data.csv'), index=False)
    for k, v in statistics.items():
        print('%s: %d' % (k, v))
    save_json(statistics, os.path.join(output_path, 'statistics.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',
                        help='input path that contains origin data')
    parser.add_argument('output_path',
                        help='output path that cleaned data is written to')
    args = parser.parse_args()
    main(args.input_path, args.output_path)
