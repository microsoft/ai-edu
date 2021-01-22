# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

class CLayer(object):
    def __init__(self, layer_type):
        self.layer_type = layer_type

    def initialize(self, folder):
        pass

    def train(self, input, train=True):
        pass

    def update(self):
        pass

    def save_parameters(self, folder, name):
        pass

    def load_parameters(self, folder, name):
        pass
