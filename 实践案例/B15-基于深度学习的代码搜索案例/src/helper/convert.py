# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import tables
from collections import OrderedDict
import os
import pandas as pd
from typing import Dict, Iterable, Any

from .file import save_pickle, save_json
from .vocabulary import build_vocab, convert_words_to_codes


def save_codenn_series(series: Iterable[str],
                       word2code: Dict[str, int],
                       file_path: str,
                       separator: str = '|') -> None:
    """Save series into file using CODEnn hdf5 format.

    Args:
        series (iterable of str): series of `sep` separated string.
        word2code (dict of str to int): word-to-code mapper.
        file_path (str): path to the output file.
        separator (str): separator to separate string into words.
    """
    with tables.open_file(file_path, mode='w') as h5f:
        table = h5f.create_table('/', 'indices', {
            'length': tables.UInt32Col(),
            'pos': tables.UInt32Col()
        }, 'a table of indices and lengths')
        array = h5f.create_earray('/', 'phrases', tables.Int32Atom(), (0,))
        array.flavor = 'numpy'
        pos = 0
        for item in series:
            item = item.split(separator)
            length = len(item)
            index = table.row
            index['length'] = length
            index['pos'] = pos
            index.append()
            array.append(convert_words_to_codes(item, word2code))
            pos += length


def save_codenn_dataset(tasks: Dict[str, pd.DataFrame],
                        max_vocab_size: int,
                        min_frequency: int,
                        output_path: str) -> Dict[str, Any]:
    """Save CODEnn dataset into multiple files for training.

    Args:
        tasks (dict of str to pandas.DataFrame): ordered dictionary contain name
                and data frame pair including `total` data frame, which is used
                for building vocabulary, each data frame should contain `name`,
                `apis`, `tokens` and `desc` column.
        max_vocab_size (int): max vocabulary size to build vocabulary.
        min_frequency (int): min appeared time for vocabulary.
        output_path (str): output path.

    Returns:
        dict of str to any: statistics object.
    """
    series = ['name', 'apis', 'tokens', 'desc']
    statistics = OrderedDict()
    word2codes = {}
    for name in series:
        data = list(map(lambda x: x.split('|'), list(tasks['total'][name])))
        word2code, code2word, original_vocab_size = build_vocab(
            data, max_vocab_size, min_frequency)
        statistics['%sOriginalVocabSize' % name] = original_vocab_size
        statistics['%sVocabSize' % name] = len(code2word)
        save_pickle(word2code, os.path.join(output_path,
                                            'word2code.%s.pkl' % name))
        save_pickle(code2word, os.path.join(output_path,
                                            'code2word.%s.pkl' % name))
        word2codes[name] = word2code
    for task_name, task in tasks.items():
        statistics['%sDatasetSize' % task_name] = task.shape[0]
        for series_name in series:
            if task_name == 'total' and series_name == 'desc':
                continue
            save_codenn_series(task[series_name], word2codes[series_name],
                               os.path.join(output_path, '%s.%s.h5' %
                                            (task_name, series_name)))
    save_json(statistics, os.path.join(output_path, 'statistics.json'))
    return statistics
