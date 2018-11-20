#!/usr/bin/env python3
# encoding: utf-8

import neural_net as nn
import sys
import numpy

if len(sys.argv) < 4:
    error_message = 'Modo de usar: \n\tpython3 main.py network.txt initial_weights.txt dataset.txt'
    print(error_message, file=sys.stderr)
    sys.exit(1)

reg_param, nodes_per_layer = nn.network_config(sys.argv[1])
weights = nn.network_weights(sys.argv[2])
dataset = nn.parse_instances(sys.argv[3])

instance = dataset[0]
net = nn.Network(weights, reg_param)

# Este método está imprimindo as coisas por enquanto.
net.train(instance)
