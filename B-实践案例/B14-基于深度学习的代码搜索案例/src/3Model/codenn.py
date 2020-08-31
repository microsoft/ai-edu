# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tensorboardX import SummaryWriter
import spacy
from typing import Any

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from helper.file import load_json, load_pickle, save_pickle
from joint_embedder import get_model, codenn_search
from dataset import get_load_dataset
from retrieval import eval_retrieval
from data_parallel import get_data_parallel
from args import get_codenn_argparser
from helper.running_log import RunningLog
from helper.helper import load_epoch
from helper.web_search import run_web_search
from helper.extractor import unescape


def normalize(data):
    """normalize matrix by rows"""
    normalized_data = data / np.linalg.norm(data, axis=1) \
        .reshape((data.shape[0], 1))
    return normalized_data


if __name__ == '__main__':
    parser = get_codenn_argparser()
    args = parser.parse_args()
    running_log = RunningLog(args.model_path)
    running_log.set('parameters', vars(args))
    os.makedirs(args.model_path, exist_ok=True)
    assert args.dataset_path is not None or args.task in ['search'], \
        '%s task requires dataset' % args.task
    assert args.load > 0 or args.task in ['train'], \
        "it's nonsense to %s on an untrained model" % args.task
    dataset_statistics = load_json(
        os.path.join(args.dataset_path, 'statistics.json'))
    model = get_data_parallel(get_model(dataset_statistics, args), args.gpu)
    if args.gpu:
        if -1 in args.gpu:
            deviceids = 'cuda'
        else:
            deviceids = 'cuda:%d' % args.gpu[0]
    else:
        deviceids = 'cpu'
    device = torch.device(deviceids)
    load_dataset = get_load_dataset(args)
    optimizer_state_dict = None
    if args.load > 0:
        model_state_dict, optimizer_state_dict = \
            load_epoch(args.model_path, args.load)
        model.load_state_dict(model_state_dict)
    model.to(device)
    running_log.set('state', 'interrupted')
    if args.task == 'train':
        train_data_loader = DataLoader(load_dataset('train'),
                                       batch_size=args.batch_size,
                                       shuffle=True, drop_last=True)
        valid_data_loader = None
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        writer = SummaryWriter(comment=args.comment or
                               os.path.basename(args.model_path))
        step = 0
        for epoch in tqdm(range(args.load + 1, args.epoch + 1), desc='Epoch'):
            losses = []
            for iter, data in enumerate(tqdm(train_data_loader, desc='Iter'),
                                        1):
                data = [x.to(device) for x in data]
                loss = model(*data).mean()
                losses.append(loss.item())
                writer.add_scalar('train/loss', loss.item(), step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if iter % args.log_every_iter == 0 or iter == len(train_data_loader)-1:
                    # noinspection PyStringFormat
                    tqdm.write('epoch:[%d/%d] iter:[%d/%d] Loss=%.5f' %
                               (epoch, args.epoch, iter, len(train_data_loader),
                                np.mean(losses)))
                    losses = []
                step += 1
            if args.valid_every_epoch and epoch % args.valid_every_epoch == 0:
                if valid_data_loader is None:
                    valid_data_loader = DataLoader(
                        load_dataset('valid'), batch_size=args.eval_pool_size,
                        shuffle=True, drop_last=True)
                model.eval()
                acc, mrr, map, ndcg = eval_retrieval(model, valid_data_loader,
                                                     device,
                                                     args.eval_pool_size,
                                                     args.eval_k)
                tqdm.write('ACC=%f, MRR=%f, MAP=%f, nDCG=%f' %
                           (acc, mrr, map, ndcg))
                writer.add_scalar('eval/acc', acc, epoch)
                writer.add_scalar('eval/mrr', mrr, epoch)
                writer.add_scalar('eval/map', map, epoch)
                writer.add_scalar('eval/ndcg', ndcg, epoch)
                model.train()
            if args.save_every_epoch and epoch % args.save_every_epoch == 0:
                tqdm.write('saving to epoch.%04d.pth' % epoch)
                torch.save((model.state_dict(), optimizer.state_dict()),
                           os.path.join(args.model_path,
                                        'epoch.%04d.pth' % epoch))
    elif args.task in ['valid', 'test']:
        model.eval()
        eval_data_loader = DataLoader(load_dataset(args.task),
                                      batch_size=args.eval_pool_size,
                                      shuffle=True, drop_last=True)
        print('ACC=%f, MRR=%f, MAP=%f, nDCG=%f' % eval_retrieval(
            model, eval_data_loader, device, args.eval_pool_size, args.eval_k))
    elif args.task == 'repr':
        model.eval()
        repr_data_loader = DataLoader(load_dataset('total'),
                                      batch_size=args.batch_size,
                                      shuffle=False, drop_last=False)
        vecs = None
        for data in tqdm(repr_data_loader, desc='Repr'):
            data = [x.to(device) for x in data]
            reprs = model.forward_code(*data).data.cpu().numpy()
            vecs = reprs if vecs is None else np.concatenate((vecs, reprs), 0)
        vecs = normalize(vecs)
        print('saving codes to use.codes.pkl')
        save_pickle(vecs, os.path.join(args.model_path, 'use.codes.pkl'))
    elif args.task == 'search':
        model.eval()
        reprs = load_pickle(os.path.join(args.model_path, 'use.codes.pkl'))
        code = pd.read_csv(os.path.join(args.dataset_path, 'use.codemap.csv'))
        assert reprs.shape[0] == code.shape[0], 'Broken data'
        word2code = load_pickle(os.path.join(args.dataset_path,
                                             'word2code.desc.pkl'))
        nlp = spacy.load('en_core_web_lg')
        while True:
            try:
                query = input('> ')
            except EOFError:
                break
            idx = codenn_search(query, model, word2code, reprs, nlp, device,
                                args.search_top_n)
            for i in idx:
                record = code.loc[i]
                if 'code' in record.index:
                    print('========')
                    print(unescape(record['code']))
                else:
                    print('==== %s:%d ====' % (record['file'], record['start']))
                    start = record['start'] - 1
                    end = record['end'] if 'end' in record else start + 10
                    with open(record['file']) as f:
                        print(''.join(f.readlines()[start:end]).strip())
    elif args.task == 'serve':
        model.eval()
        reprs = load_pickle(os.path.join(args.model_path, 'use.codes.pkl'))
        code = pd.read_csv(os.path.join(args.dataset_path, 'use.codemap.csv'), na_filter=False)
        assert reprs.shape[0] == code.shape[0], 'Broken data'
        word2code = load_pickle(os.path.join(args.dataset_path,
                                             'word2code.desc.pkl'))
        nlp = spacy.load('en_core_web_lg')

        def handler(query: str, count: int) -> Any:
            results = []
            idx = codenn_search(query, model, word2code, reprs, nlp, device,
                                count)
            for i in idx:
                record = code.loc[i]
                if 'code' in record.index:
                    if 'url' in record.index:
                        results.append({
                            'code': unescape(record['code']),
                            'url': record['url']
                        })
                    else:
                        results.append({
                            'code': unescape(record['code'])
                        })
                else:
                    start = record['start'] - 1
                    end = record['end'] if 'end' in record else start + 10
                    with open(record['file']) as f:
                        results.append({
                            'code': ''.join(f.readlines()[start:end]).strip()
                        })
            return results
        run_web_search(args.host, args.port, handler)
    running_log.set('state', 'succeeded')
