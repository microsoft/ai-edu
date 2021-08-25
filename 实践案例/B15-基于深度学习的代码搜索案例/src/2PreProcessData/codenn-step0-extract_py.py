# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import argparse
from tqdm import tqdm
from collections import OrderedDict

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helper.file import find_all_files_with_extension, save_json
from helper.extractor import extract_python_for_codenn, escape


def main(input_path: str, output_path: str) -> None:
    statistics = OrderedDict()
    files = list(find_all_files_with_extension(input_path, '.py'))
    statistics['filesNum'] = len(files)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'data.csv'), "w") as out:
        f = csv.writer(out)
        f.writerow(['file', 'start', 'name', 'apis', 'tokens',
                    'desc', 'imported', 'code', 'url'])
        records_num = 0
        for file in tqdm(files, desc='Extract'):
            try:
                with open(file) as source:
                    content = source.read()
                functions, packages = extract_python_for_codenn(content)
                records_num += len(functions)
                for function in functions:
                    name, line, apis, tokens, desc, code = function
                    url = generateGithubUrl(input_path, file, line)
                    f.writerow((file, line, name,
                                '|'.join(apis),
                                '|'.join(tokens),
                                escape(desc or ''),
                                '|'.join(packages),
                                escape(code),
                                url))
            except (SyntaxError, UnicodeDecodeError,
                    UnicodeEncodeError, MemoryError):
                pass
    statistics['recordsNum'] = records_num
    for k, v in statistics.items():
        print('%s: %d' % (k, v))
    save_json(statistics, os.path.join(output_path, 'statistics.json'))


def generateGithubUrl(input_path, file, line):
    basepath = os.path.normcase(input_path)
    localpath = os.path.normcase(file)
    paths = list(filter(None, localpath.replace(basepath, '', 1).split(os.sep)))
    paths.insert(0, 'https://github.com')
    paths.insert(3, 'blob') 
    return '/'.join(paths)+'#L'+str(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',
                        help='input path that contains raw python code')
    parser.add_argument('output_path',
                        help='output path that extracted data is written to')
    args = parser.parse_args()
    main(args.input_path, args.output_path)
