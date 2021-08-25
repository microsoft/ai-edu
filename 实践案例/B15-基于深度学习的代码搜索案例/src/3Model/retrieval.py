# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import dataloader
from tqdm import tqdm
from typing import List, Tuple

from choices import similarity_choices


def ACC(real: List[int], predict: List[int]) -> float:
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum + 1
    return sum / float(len(real))


def MAP(real: List[int], predict: List[int]) -> float:
    sum = 0.0
    for id, val in enumerate(real):
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum + (id + 1) / float(index + 1)
    return sum / float(len(real))


def MRR(real: List[int], predict: List[int]) -> float:
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum + 1.0 / float(index + 1)
    return sum / float(len(real))


def NDCG(real: List[int], predict: List[int]) -> float:
    dcg = 0.0
    idcg = IDCG(len(real))
    for i, predictItem in enumerate(predict):
        if predictItem in real:
            item_relevance = 1
            rank = i + 1
            dcg += (math.pow(2, item_relevance) - 1.0) * (math.log(2) / math.log(rank + 1))
    return dcg / float(idcg)


def IDCG(n: int) -> float:
    idcg = 0
    item_relevance = 1
    for i in range(n):
        idcg += (math.pow(2, item_relevance) - 1.0) * (math.log(2) / math.log(i + 2))
    return idcg


def eval_retrieval(model: nn.Module, data_loader: dataloader.DataLoader,
                   device: torch.device, pool_size: int, k: int,
                   similarity: str = 'cos') -> Tuple[float, float, float, float]:
    """Evaluate retrieval score on model with a progress bar.
    
    Args:
        model (torch.nn.Module): model object, you should call eval() before
                calling this function.
        data_loader (torch.utils.data.dataloader.Dataloader): evaluation dataset
                loader object, it's batch size should be set to `pool_size`.
        device (torch.device): device object.
        pool_size (int): pool size.
        k (int): top k search.
        similarity (str): `cos` or `l2`.
    
    Returns:
        tuple: tuple containing:
            float: ACC index.
            float: MRR index.
            float: MAP index.
            float: nD
    """
    assert data_loader.batch_size == pool_size, \
        'Pool size must equal to batch size'
    accs, mrrs, maps, ndcgs = [], [], [], []
    for names, apis, tokens, descs, _ in tqdm(data_loader, desc='Eval'):
        names, apis, tokens, descs = [tensor.to(device) for tensor in
                                      (names, apis, tokens, descs)]
        code_repr = model.forward_code(names, apis, tokens)
        descs_repr = model.forward_desc(descs)
        for i in range(pool_size):
            desc_repr = descs_repr[i].expand(pool_size, -1)
            sims = similarity_choices[similarity](code_repr, desc_repr) \
                .data.cpu().numpy()
            predict = np.argsort(sims)
            predict = predict[:k]
            predict = [int(k) for k in predict]
            real = [i]
            accs.append(ACC(real, predict))
            mrrs.append(MRR(real, predict))
            maps.append(MAP(real, predict))
            ndcgs.append(NDCG(real, predict))
    # noinspection PyTypeChecker
    return np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)
