# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from Level0_OverFitNet import *

if __name__ == '__main__':

    dr = LoadData()

    num_input = dr.num_feature
    num_hidden = 64
    num_output = 1
    max_epoch = 10000
    batch_size = 5
    learning_rate = 0.1
    eps = 1e-6

    params = HyperParameters41(
        learning_rate, max_epoch, batch_size, eps,                        
        net_type=NetType.Fitting,
        init_method=InitialMethod.Xavier, 
        optimizer_name=OptimizerName.SGD,
        regular_name=RegularMethod.L2, regular_value=0.005)

    net = Model(dr, num_input, num_hidden, num_output, params)
    ShowResult(net, dr)
