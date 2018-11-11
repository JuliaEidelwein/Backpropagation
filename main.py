# encoding: utf-8

# from collections import namedtuple
# import sys
# import csv
# import random
# import test_and_training as tat

import sys
from collections import namedtuple

import neural_net as nn

def parse_instances(filename):
    dataset = []
    with open(filename) as file:
        next(file)
        for line in file:
            substring = line.split(';')
            data = tuple((float(s) for s in substring[0].split(',')))
            result =  tuple((float(s) for s in substring[1].split(',')))
            dataset.append((data,result))
    return dataset

def normalize_dataset(D):
    maxV = []
    minV = []
    for attr in range(len(D[0][0])):
        maxV.append(max(D[i][0][attr] for i in range(len(D))))
        minV.append(min(D[i][0][attr] for i in range(len(D))))
    normalizedDataset = []
    for instance in D:
        normalizedAttr = []
        for attr in range(len(instance[0])):
            normalizedAttr.append((((instance[0][attr] - minV[attr])*2)/(maxV[attr] - minV[attr])) - 1)
        normalizedDataset.append((normalizedAttr,instance[1]))
    print(maxV)
    print(minV)
    return normalizedDataset, minV, maxV

if __name__ == '__main__':

    if len(sys.argv) < 4:
        print('Número de argumentos insuficiente', file=sys.stderr)
        sys.exit(1)

    network_filename = sys.argv[1]
    initial_weights_filename = sys.argv[2]
    dataset_filename = sys.argv[3]

    reg_param, nodes_per_layer = nn.network_config(network_filename)
    weights = nn.network_weights(initial_weights_filename)
    dataset = parse_instances(dataset_filename)
    print(weights)
    print(dataset)
    normalizedDataset, maxV, minV = normalize_dataset(dataset)
    print(normalizedDataset)

