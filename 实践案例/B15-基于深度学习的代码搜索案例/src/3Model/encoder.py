# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.functional as F

from choices import pool_choices, rnn_choices, activations_choices


class BOWEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, pool='max', activation='tanh'):
        super(BOWEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        assert pool in pool_choices.keys(), 'Invalid pool option'
        self.pool = pool_choices[pool]
        assert activation in activations_choices.keys(), \
            'Invalid activation option'
        self.activation = activations_choices[activation]

    def forward(self, input):
        embedded = F.dropout(self.embedding(input), 0.25, self.training)
        return self.activation(self.pool(embedded, dim=1))


class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, rnn='lstm',
                 bidirectional=True, pool='max', activation='tanh'):
        super(SeqEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        assert rnn in rnn_choices.keys(), 'Invalid RNN option'
        self.rnn = rnn_choices[rnn](embed_size, hidden_size,
                                    batch_first=True,
                                    bidirectional=bidirectional)
        for w in self.rnn.parameters():
            if w.dim() > 1:
                weight_init.orthogonal_(w)
        assert pool in pool_choices.keys(), 'Invalid pool option'
        self.pool = pool_choices[pool]
        assert activation in activations_choices.keys(), \
            'Invalid activation option'
        self.activation = activations_choices[activation]

    def forward(self, input):
        embedded = F.dropout(self.embedding(input.long()), 0.25, self.training)
        self.rnn.flatten_parameters()
        rnn_output = F.dropout(self.rnn(embedded)[0], 0.25, self.training)
        return self.activation(self.pool(rnn_output, dim=1))
