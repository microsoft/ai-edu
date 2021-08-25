# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from collections import Counter
from typing import List, Union, Tuple, Dict, Iterable, Any
from itertools import chain
import pandas as pd


def build_vocab(texts: Iterable[List[str]],
                max_vocab_size: Union[int, None] = None,
                min_frequency: int = 2,
                reserved: Union[List[str], None] = None) -> \
        Tuple[Dict[str, int], List[str], int]:
    """Create vocabulary from list of texts.

    Args:
        texts (iterable of list of str): list of tokenized text.
        max_vocab_size (int or None): max vocabulary size including reserved
                words, pass None (default) to mean no limit.
        min_frequency (int): min frequency a word appear.
        reserved (list of str or None): reserved word placed which has the
                lowest code, default is `<PAD>` and '<UNK>'.

    Returns:
        tuple: tuple containing:
            dict of str to int: a dictionary map word to code.
            list of str: a list map code to word.
            int: original vocabulary size.
    """
    if reserved is None:
        reserved = ['<PAD>', '<UNK>']
    counter = Counter(chain(*texts))
    original_vocab_size = len(counter)
    counter_items = list(filter(
        lambda x: x[1] >= min_frequency and x[0] not in reserved,
        counter.items()))
    counter_items.sort(key=lambda x: x[1], reverse=True)
    code2word = list(map(lambda x: x[0], counter_items))
    if max_vocab_size is None:
        code2word = reserved + code2word
    else:
        code2word = reserved + code2word[:max_vocab_size - len(reserved)]
    return ({v: k for k, v in enumerate(code2word)},
            code2word, original_vocab_size)


def convert_words_to_codes(words: List[str], word2code: Dict[str, int]) \
        -> List[int]:
    """Convert list of words to list of codes.

    Args:
        words (list of str): words to convert.
        word2code (dict of str to int): dictionary mapping word to code.

    Returns:
        list of int: list of converted codes.
    """
    unknown_code = word2code['<UNK>']
    return list(map(lambda x: word2code.get(x, unknown_code), words))


def heuristic_sequence_length(series: Iterable[List[Any]],
                              percent: float = 0.9,
                              bin: int = 5) -> int:
    """Pick suitable sequence according to percentage.

    Args:
        series (iterable of list of any): list of list.
        percent (float): percentage of data the result length can hold.
        bin (int): length of the list in series is round to nearest `n * bin`.

    Returns:
        int: suitable sequence length.
    """
    series_length = map(lambda x: int(bin * round(float(len(x)) / bin)), series)
    histogram = pd.DataFrame(Counter(series_length).items(),
                             columns=['bin', 'count']).sort_values(by='bin')
    histogram['cumsum_percent'] = histogram['count'].cumsum() / \
                                  histogram['count'].sum()
    return int(histogram[histogram.cumsum_percent >= percent].bin.iloc[0])
