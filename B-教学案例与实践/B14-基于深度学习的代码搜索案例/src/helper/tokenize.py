# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import re
from nltk.tokenize import RegexpTokenizer
from spacy.language import Language
from typing import List, Callable
import itertools
import spacy
import pandas as pd
from unidecode import unidecode
from textacy.preprocessing.normalize import normalize_unicode, \
    normalize_whitespace
from textacy.preprocessing.replace import replace_urls, replace_emails, \
    replace_phone_numbers, replace_numbers, replace_currency_symbols
from textacy.preprocessing.remove import remove_accents, remove_punctuation
from keras_preprocessing.text import text_to_word_sequence

cop_regex = re.compile("[^a-z0-9]")
split_name_regex = re.compile(r'(.+?(?:(?<=[a-z])(?=[A-Z])|'
                              r'(?<=[A-Z])(?=[A-Z][a-z])|$))')
stop_words = ["a", "about", "above", "after", "again", "against", "ain", "all",
              "am", "an", "and", "any", "are", "aren", "aren't", "as", "at",
              "be", "because", "been", "before", "being", "below", "between",
              "both", "but", "by", "can", "couldn", "couldn't", "d", "did",
              "didn", "didn't", "do", "does", "doesn", "doesn't", "doing",
              "don", "don't", "down", "during", "each", "few", "for", "from",
              "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't",
              "have", "haven", "haven't", "having", "he", "her", "here",
              "hers", "herself", "him", "himself", "his", "how", "i", "if",
              "in", "into", "is", "isn", "isn't", "it", "it's", "its",
              "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't",
              "more", "most", "mustn", "mustn't", "my", "myself", "needn",
              "needn't", "no", "nor", "not", "now", "o", "of", "off", "on",
              "once", "only", "or", "other", "our", "ours", "ourselves", "out",
              "over", "own", "re", "s", "same", "shan", "shan't", "she",
              "she's", "should", "should've", "shouldn", "shouldn't", "so",
              "some", "such", "t", "than", "that", "that'll", "the", "their",
              "theirs", "them", "themselves", "then", "there", "these", "they",
              "this", "those", "through", "to", "too", "under", "until", "up",
              "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren",
              "weren't", "what", "when", "where", "which", "while", "who",
              "whom", "why", "will", "with", "won", "won't", "wouldn",
              "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've",
              "your", "yours", "yourself", "yourselves", "could", "he'd",
              "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've",
              "let's", "ought", "she'd", "she'll", "that's", "there's",
              "they'd", "they'll", "they're", "they've", "we'd", "we'll",
              "we're", "we've", "what's", "when's", "where's", "who's",
              "why's", "would"]


def get_language(name: str = 'en_core_web_lg') -> Language:
    """Return configured spacy language object.

    Args:
        name (str): model name.

    Returns:
        spacy.language.Language: language model.
    """
    nlp = spacy.load('en_core_web_lg')
    for stopword in stop_words:
        nlp.vocab[stopword].is_stop = True
    return nlp


def tokenize_docstring_for_baseline(text: str, nlp: Language) -> List[str]:
    """Apply tokenization using spacy to docstrings for baseline.

    Args:
        text (str): docstring.
        nlp (spacy.language.Language): spacy language object.

    Returns:
        list of str: list of tokens. Whitespace and stop words are removed.
    """
    all_tokens = nlp.tokenizer(text.lower())
    selected_tokens = [cop_regex.sub('', token.text) for token in all_tokens
                       if not token.is_space and not token.is_stop]
    return [token for token in selected_tokens if token != '']


def tokenize_docstring_for_codenn(text: str, nlp: Language) -> List[str]:
    """Apply tokenization using spacy to docstrings for codenn.

    Args:
        text (str): docstring.
        nlp (spacy.language.Language): spacy language object.

    Returns:
        list of str: list of tokens. Only non-stop alpha word is kept.
    """
    return [token.lemma_ for token in nlp(text)
            if token.is_alpha and not token.is_stop]


def tokenize_code(text: str) -> List[str]:
    """A very basic procedure for tokenizing code strings.

    Args:
        text (str): source code.

    Returns:
        list of str: list of tokens.
    """
    return RegexpTokenizer(r'\w+').tokenize(text)


def split_name(name: str) -> List[str]:
    """Split name according to underscore and camelcase naming rules.

    Args:
        name (str): name to split.

    Returns:
        list of str: list of splitted names.
    """
    matches = re.finditer(split_name_regex, name)
    return list(itertools.chain(
        *[[i for i in m.group(0).split('_') if i] for m in matches]))


line_break_regex = re.compile(r'((\r\n)|[\n\v])+')
non_breaking_space_regex = re.compile(r'(?!\n)\s+')


def default_cleaner(text: str, fix_unicode: bool = True, lowercase: bool = True,
                    transliterate: bool = True, no_urls: bool = True,
                    no_emails: bool = True, no_phone_numbers: bool = True,
                    no_numbers: bool = True, no_currency_symbols: bool = True,
                    no_punct: bool = True, no_accents: bool = True) -> str:
    """Default function to clean text."""
    if fix_unicode:
        text = normalize_unicode(text, form='NFC')
    if transliterate is True:
        text = unidecode(text)
    if lowercase is True:
        text = text.lower()
    if no_urls:
        text = replace_urls(text, '<URL>')
    if no_emails is True:
        text = replace_emails(text, '<EMAIL>')
    if no_phone_numbers is True:
        text = replace_phone_numbers(text, '<PHONE>')
    if no_numbers is True:
        text = replace_numbers(text, '<NUMBER>')
    if no_currency_symbols is True:
        text = replace_currency_symbols(text, '<CUR>')
    if no_accents is True:
        text = remove_accents(text)
    if no_punct is True:
        text = remove_punctuation(text)
    return normalize_whitespace(text)


def default_tokenizer(text: str) -> List[str]:
    """Default function to tokenize text."""
    return text_to_word_sequence(text, lower=False,
                                 filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')


def process_text(series: pd.Series,
                 cleaner: Callable[[str], str] = default_cleaner,
                 tokenizer: Callable[[str], List[str]] = default_tokenizer,
                 append_indicators: bool = False,
                 tqdm=False) -> pd.Series:
    """Process text with cleaner and tokenizer.

    Args:
        series (pandas.Series of str): series of str to process.
        cleaner (function): str to str cleaning function.
        tokenizer (function): str to list of str tokenizing function.
        append_indicators (bool): whether prepend SOS and append EOS to
                tokenized text.
        tqdm (bool): whether use tqdm progress bar.

    Returns:
        pandas.Series of list of str: series of list of words.
    """
    func = 'progress_map' if tqdm else 'map'
    if append_indicators:
        return getattr(series, func)(
            lambda x: ['<SOS>'] + tokenizer(cleaner(x)) + ['<EOS>'])
    return getattr(series, func)(lambda x: tokenizer(cleaner(x)))
