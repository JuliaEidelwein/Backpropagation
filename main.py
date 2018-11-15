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
# normalized_dataset, max_values, min_values = nn.normalize_dataset(dataset)
normalized_dataset = dataset

print('>> Lambda: ', reg_param, end="\n\n"),
print('>> Nodos por camada: ', nodes_per_layer, end="\n\n")
print('>> Pesos iniciais lidos: ', weights, end="\n\n")
print('>> Dados lidos', dataset, end="\n\n")
print('>> Dados normalizados', normalized_dataset, end="\n\n")
print('>> Pesos', weights)

instance = dataset[0]
z, activations = nn.propagate(weights, instance)

print(">> Z: ", z, end="\n\n")
print(">> Ativações: ", activations, end="\n\n")

errors = nn.J(activations[-1], instance.result)
print(">> Saída predita para o exemplo:", activations[-1], end="\n\n")
print(">> Valor experado para o exemplo:", instance.result, end="\n\n")
print(">> Erros obtidos:", errors, end="\n\n")
print(nn.sigmoid(numpy.asarray([1, 2, 3])), end="\n\n")

nn.backpropagation(weights, dataset, nodes_per_layer)
