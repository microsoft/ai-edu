# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import argparse


def add_common_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments like `--dataset_path`, `--model_path`, `--gpu`,
    `--load`, `--epoch`, `--batch_size`, `--learning_rate`, `--log_every_iter`,
    `--valid_every_epoch`, `--save_every_epoch`, `--comment`."""
    parser.add_argument('--dataset_path', help='path to the dataset')
    parser.add_argument('--model_path', help='path for saving models and codes',
                        required=True)
    parser.add_argument('--gpu', type=lambda x: list(map(int, x.split(','))),
                        default=[], nargs='?', const=[-1],
                        help='GPU ids splited by ",", no-ids to use all GPU')
    parser.add_argument('--load', type=int, default=0,
                        help='load module training at give epoch')
    parser.add_argument('--epoch', type=int, default=200, help='epoch to train')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--log_every_iter', type=int, default=100,
                        help='log loss every numbers of iteration')
    parser.add_argument('--valid_every_epoch', type=int, default=10,
                        help='run validation every numbers of epoch; '
                             '0 for disabling')
    parser.add_argument('--save_every_epoch', type=int, default=10,
                        help='save model every numbers of epoch; '
                             '0 for disabling')
    parser.add_argument('--comment', default='', help='comment for tensorboard')


def add_server_arguments(parser: argparse.ArgumentParser) -> None:
    """Add server arguments like `--port` and `--host`."""
    parser.add_argument('--port', type=int, default=8080,
                        help='port to serve at')
    parser.add_argument('--host', default='0.0.0.0', help='address to serve at')


def add_retrieval_eval_arguments(parser: argparse.ArgumentParser) -> None:
    """Add retrieval evaluation arguments like `--eval_pool_size` and
    `--eval_k`."""
    parser.add_argument('--eval_pool_size', type=int, default=200,
                        help='pool size for evaluation')
    parser.add_argument('--eval_k', type=int, default=10,
                        help='k for evaluation')


def add_search_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--search_top_n', type=int, default=5,
                        help='search top-n results for search task')


def add_codenn_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--embed_size', type=int, default=100,
                        help='embedding size')
    parser.add_argument('--repr_size', type=int, default=100,
                        help='representation size;  for bidirectional rnn, the '
                             'real value will be doubled')
    parser.add_argument('--pool', choices=['max', 'mean', 'sum'], default='mean',
                        help='pooling method to use')
    parser.add_argument('--rnn', choices=['lstm', 'gru', 'rnn'], default='gru',
                        help='rnn and rnn variants to use')
    parser.add_argument('--bidirectional', choices=['true', 'false'],
                        default='true', help='whether to use bidirectional rnn')
    parser.add_argument('--activation', choices=['relu', 'tanh'],
                        default='relu', help='activation function to use')
    parser.add_argument('--margin', type=float, default=0.05,
                        help='margin to use in the loss function')
    parser.add_argument('--name_len', type=int, default=6,
                        help='length of name sequence')
    parser.add_argument('--api_len', type=int, default=30,
                        help='length of api sequence')
    parser.add_argument('--token_len', type=int, default=50,
                        help='length of tokens')
    parser.add_argument('--desc_len', type=int, default=30,
                        help='length of description sequence')


def add_code_summarizer_train_arguments(parser: argparse.ArgumentParser) \
        -> None:
    parser.add_argument('--embed_size', type=int, default=800,
                        help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=1000,
                        help='hidden state size')


def get_codenn_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',
                        choices=['train', 'valid', 'test', 'repr',
                                 'search', 'serve'],
                        default='train',
                        help="task to run; `train' for training the dataset; "
                             "`valid'/`test' for evaluating model on "
                             "corresponding dataset; `repr' for converting "
                             "whole dataset(`use') to code; `search' for "
                             "searching in whole dataset, it require `repr' to "
                             "run first; `serve' for searching as web server, "
                             "it require `repr' to run first")
    add_common_model_arguments(parser)
    add_codenn_train_arguments(parser)
    add_retrieval_eval_arguments(parser)
    add_search_arguments(parser)
    add_server_arguments(parser)
    return parser


def get_code_summarizer_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',
                        choices=['train', 'valid', 'test', 'summarize'],
                        default='train',
                        help="task to run; `train' for training the dataset; "
                             "`valid'/`test' for evaluating model on "
                             "corresponding dataset; `summarize' for summarize "
                             "the user input code")
    add_common_model_arguments(parser)
    add_code_summarizer_train_arguments(parser)
    return parser
