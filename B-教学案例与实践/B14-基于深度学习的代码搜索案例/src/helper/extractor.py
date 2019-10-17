# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import ast
from collections import deque
from typing import List, Tuple
import astor
import json
from tqdm import tqdm
from spacy.language import Language

from .tokenize import tokenize_docstring_for_baseline, tokenize_code


class CallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.names = deque()

    def visit_Name(self, node: ast.Name):
        self.names.appendleft(node.id)

    def visit_Attribute(self, node: ast.Attribute):
        try:
            self.names.appendleft(node.attr)
            self.names.appendleft(node.value.id)
        except AttributeError:
            self.generic_visit(node)


def extract_python_for_codenn(content: str) -> \
        Tuple[List[Tuple[str, int, List[str], List[str], str, str]], List[str]]:
    """Extract signature for python module.

    Args:
        content (str): the content to extract. It must be a valid python source
            code, or exception may be thrown.

    Returns:
        tuple: tuple containing:
            list of tuple: tuple containing:
                str: function name.
                int: starting line number of the code.
                list of str: api sequence each of which is `.` separated.
                list of str: token sequence including names and attributes.
                str: docstring of the function.
                str: source code of the function.
            list of str: list of imported package names, each of which  is `.`
                    separated.
    """
    module = ast.parse(content)
    functions = []
    packages = []
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef):
            name = node.name
            api = []
            token = []
            for sub_node in ast.walk(node):
                if isinstance(sub_node, ast.Name):
                    token.append(sub_node.id)
                elif isinstance(sub_node, ast.Attribute):
                    token.append(sub_node.attr)
                elif isinstance(sub_node, ast.Call):
                    visitor = CallVisitor()
                    visitor.visit(sub_node.func)
                    api.append('.'.join(visitor.names))
            desc = ast.get_docstring(node)
            code = astor.to_source(node)
            functions.append((name, node.lineno, api, token, desc, code))
        if isinstance(node, ast.Import):
            for name in node.names:
                packages.append(name.name)
    return functions, packages


def extract_python_for_baseline(content: str) -> \
        List[Tuple[str, int, str, str, str]]:
    """Extract (function/method, docstring) pairs from given source code.

    Args:
        content (str): the content to extract. It must be a valid python source
            code, or exception may be thrown.

    Returns:
        list of tuple: tuple containing:
            str: name of global function or method, if it's a method, class name
                    and a dot is prepended.
            int: start line number.
            str: source code generated from AST (maybe different from original
                    code).
            str: function of which docstring is removed
            str: docstring of function, empty string if no docstring found.
    """
    results = []
    method2class = {}
    module = ast.parse(content)
    classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
    functions = [node for node in module.body if isinstance(node,
                                                            ast.FunctionDef)]
    for _class in classes:
        for node in _class.body:
            if isinstance(node, ast.FunctionDef):
                functions.append(node)
                method2class[node] = _class.name
    for f in functions:
        source = astor.to_source(f)
        docstring = ast.get_docstring(f) if ast.get_docstring(f) else ''
        class_name = method2class.get(f)
        results.append((
            class_name + '.' + f.name if class_name else f.name,
            f.lineno,
            source,
            source.replace(ast.get_docstring(f, clean=False),
                           '') if docstring else source,
            docstring
        ))
    return results


def extract_and_tokenize_for_baseline(sources: List[str], nlp: Language) \
        -> List[List[Tuple[str, int, str, str, str]]]:
    """Extract and tokenize a list of files.

    Args:
        sources (list of str): list of source code to extract and tokenize.
        nlp (spacy.language.Language): spacy language object.

    Returns:
        list of list of tuple: tuple containing:
            str: name of global function or method, if it's a method, class name
                    and a dot is prepended.
            int: start line number.
            str: source code generated from AST (maybe different from original
                    code).
            str: tokens separated by space of source code excluding docstring.
            str: tokens separated by space of docstring.
    """
    results = []
    for source in tqdm(sources, desc='Clean'):
        tokenized = []
        try:
            untokenized = extract_python_for_baseline(source)
            tokenized = list(map(lambda pair: (
                pair[0], pair[1], pair[2],
                ' '.join(tokenize_code(pair[3])),
                ' '.join(tokenize_docstring_for_baseline(pair[4], nlp))
            ), untokenized))
        except (SyntaxError, UnicodeEncodeError, MemoryError):
            pass
        results.append(tokenized)
    return results


def escape(content: str) -> str:
    """Escape string using `json`.

    Note:
        Use `json.dumps` instead of `repr` to keep compatible with JavaScript
        extractor, which has only `JSON` library.

    Args:
        content (str): string to escape.

    Returns:
        str: escaped result.
    """
    return json.dumps(content)[1:-1]


def unescape(content: str) -> str:
    """Unescape string using `json`

    Args:
        content (str): string to unescape.

    Returns:
        str: unescaped result.
    """
    return json.loads('"' + content + '"')
