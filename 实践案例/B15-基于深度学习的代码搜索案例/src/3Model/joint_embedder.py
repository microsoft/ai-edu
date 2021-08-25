# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from typing import Dict, Any, List
from spacy.language import Language
import numpy as np

from encoder import BOWEncoder, SeqEncoder


class JointEmbedder(nn.Module):
    def __init__(self, name_vocab_size, apis_vocab_size, tokens_vocab_size,
                 desc_vocab_size, embed_size, repr_size, pool='max', rnn='lstm',
                 bidirectional=True, activation='tanh', margin=0.05):
        super(JointEmbedder, self).__init__()
        self.name_encoder = SeqEncoder(name_vocab_size, embed_size, repr_size,
                                       rnn=rnn, bidirectional=bidirectional,
                                       pool=pool, activation=activation)
        self.apis_encoder = SeqEncoder(apis_vocab_size, embed_size, repr_size,
                                      rnn=rnn, bidirectional=bidirectional,
                                      pool=pool, activation=activation)
        self.tokens_encoder = BOWEncoder(tokens_vocab_size, embed_size,
                                        pool=pool, activation=activation)
        self.desc_encoder = SeqEncoder(desc_vocab_size, embed_size, repr_size,
                                       rnn=rnn, bidirectional=bidirectional,
                                       pool=pool, activation='tanh')
        if bidirectional:
            self.fuse = nn.Linear(embed_size + 4 * repr_size, 2 * repr_size)
        else:
            self.fuse = nn.Linear(embed_size + 2 * repr_size, repr_size)
        self.margin = margin

    def forward_code(self, name, apis, tokens):
        name_repr = self.name_encoder(name)
        apis_repr = self.apis_encoder(apis)
        tokens_repr = self.tokens_encoder(tokens)
        code_repr = self.fuse(torch.cat((name_repr, apis_repr, tokens_repr), 1))
        return torch.tanh(code_repr)

    def forward_desc(self, desc):
        return self.desc_encoder(desc)

    def forward(self, name, apis, tokens, desc_good, desc_bad):
        code_repr = self.forward_code(name, apis, tokens)
        good_sim = F.cosine_similarity(code_repr, self.forward_desc(desc_good))
        bad_sim = F.cosine_similarity(code_repr, self.forward_desc(desc_bad))
        return (self.margin - good_sim + bad_sim).clamp(min=1e-6)


def codenn_search(query: str, model: nn.Module, word2code: Dict[str, int],
                  reprs: torch.Tensor, nlp: Language, device: torch.device,
                  top_n: int) \
        -> List[int]:
    words = [word2code.get(token.lemma_, 1) for token in nlp(query)
             if token.is_alpha and not token.is_stop]
    if not words:
        return []
    desc = torch.from_numpy(np.expand_dims(np.array(words), axis=0))
    desc = desc.to(device)
    desc_repr = model.forward_desc(desc).data.cpu().numpy()
    sim = np.negative(np.dot(reprs, desc_repr.transpose()).squeeze(axis=1))
    idx = np.argsort(sim)[:top_n]
    return list(idx)


def get_model(dataset_statistics: Dict[str, Any], args: argparse.Namespace) \
        -> nn.Module:
    return JointEmbedder(dataset_statistics['nameVocabSize'],
                         dataset_statistics['apisVocabSize'],
                         dataset_statistics['tokensVocabSize'],
                         dataset_statistics['descVocabSize'],
                         args.embed_size, args.repr_size,
                         args.pool, args.rnn, args.bidirectional == 'true',
                         args.activation, args.margin)
