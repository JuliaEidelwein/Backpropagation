#!/usr/bin/env python3
# encoding: utf-8

import neural_net as nn
import test_and_training as tt
import sys
import numpy

if len(sys.argv) < 4:
    error_message = 'Modo de usar: \n\tpython3 main.py network.txt initial_weights.txt dataset.txt'
    print(error_message, file=sys.stderr)
    sys.exit(1)

reg_param, nodes_per_layer = nn.network_config(sys.argv[1])
weights = nn.network_weights(sys.argv[2])
dataset = nn.parse_instances(sys.argv[3])
dataset, max_value, min_value = nn.normalize_dataset(dataset)

print("Rede " + str(nodes_per_layer) + " com lambda " + str(reg_param))

# instance = dataset[1]


validation = tt.cross_validation(weights, reg_param, dataset, k=10)
print(validation)

# net = nn.Network(weights, reg_param)
# net.train([instance])
# net.train(dataset)
# kfolds = tt.stratified_k_fold(dataset)
# for fold in kfolds:
    # for instance
    # net.train(fold)
# for instance in dataset:
    # print(instance.klass)
    # print(net.predict_class(instance))
    # print()

# Este método está imprimindo as coisas por enquanto.
# net.train(dataset, 10)
# for instance in dataset:
#     z, a = net.activate(instance.data)
#     print(instance.result, a[-1][1:], sep="\n", end="\n\n")

# EPS = 0.0000010000
# grad = net.gradients
# numeric = net.numerical_gradient_estimation(EPS, dataset)
# print("Gradiente")
# print(grad)
# print("Gradiente numérico")
# print(numeric)
# for n, g in zip(numeric, grad):
#     result = nn.error(n, g)
#     print("{0:.20f}".format(result))
