#!/usr/bin/env python3
# encoding: utf-8

import neural_net as nn
import sys
import numpy
from statistics import mean


def norm(vector):
    return numpy.linalg.norm(vector)


def check_gradient(numerical, backprop):
    # TODO: Resolver este pepino
    #error = norm(numerical - backprop) / norm(numerical + backprop)
    error = norm(numpy.asarray(numerical - backprop)) / \
        norm(numpy.asarray(numerical + backprop))
    return error


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
net.train(dataset)
print("Gradientes")
print(net.gradients)
eps = 0.0000010000
numerical = net.numerical_gradient_estimation(eps, dataset)

print("Numerical")
print(numerical)

print("Error")
for g, n in zip(net.gradients, numerical):
    print(check_gradient(n, g))
