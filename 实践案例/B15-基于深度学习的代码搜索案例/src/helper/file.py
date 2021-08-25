# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import os
from typing import Iterator, Any, Union
import json
import pickle


def find_all_files(directory: str, follow_symlinks=False) -> Iterator[str]:
    """Find all regular files recursively in directory.

    Args:
        directory (str): directory to recursively search in.
        follow_symlinks (bool): whether to follow symlinks, defaults to not.

    Yields:
        str: regular file path relative to `directory`.
    """
    for entry in os.scandir(directory):
        name = os.path.normcase(os.path.normpath(os.path.join(directory, entry.name)))
        if entry.is_dir(follow_symlinks=follow_symlinks):
            yield from find_all_files(name)
        elif entry.is_file(follow_symlinks=follow_symlinks):
            yield name


def find_all_files_with_extension(directory: str,
                                  extension: str,
                                  follow_symlinks=False) \
        -> Iterator[str]:
    """Find all regular files with extension recursively in directory.

    Args:
        directory (str): directory to recursively search in.
        extension (str): extension for searching files, it should start with
                `.`.
        follow_symlinks (bool): whether to follow symlinks, defaults to not.

    Returns:
        iterator of str: regular file path with extension relative to
                `directory`.
    """
    return filter(lambda x: x.endswith(extension),
                  find_all_files(directory, follow_symlinks))


def find_files_with_extension(directory: str, extension: str) -> Iterator[str]:
    """Find all regular files with extension in directory.

    Args:
        directory (str): directory to search in.
        extension (str): extension for searching files, it should start with
                `.`.

    Returns:
        iterator of str: regular file path with extension relative to
                `directory`.
    """
    return filter(lambda x: x.endswith(extension),
                  map(lambda x: os.path.join(directory, x),
                      os.listdir(directory)))


def save_json(obj: Any, path: str, indent: Union[int, None] = 2) -> None:
    """Save json file.

    Args:
        obj (any): object to dump.
        path (file): file path.
        indent (int or None): indent size,
    """
    with open(path, 'w') as f:
        json.dump(obj, f, indent=indent)


def load_json(path: str) -> Any:
    """Load json file.

    Args:
        path (file): file path.

    Returns:
        any: dumped object.
    """
    with open(path) as f:
        return json.load(f)


def save_pickle(obj: Any, path: str) -> None:
    """Save pickle file.

    Args:
        obj (any): object to dump.
        path (file): file path.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    """Load pickle file.

    Args:
        path (file): file path.

    Returns:
        any: dumped object.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
