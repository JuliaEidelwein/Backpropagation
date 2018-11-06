import numpy as np
import math



def network_config(filename):
    with open(filename) as file:
        regularization_parameter = float(next(file))
        nodes_per_layer = []
        for line in file:
            nodes_per_layer.append(int(line))
        return (regularization_parameter, nodes_per_layer)


def network_weights(filename):
    w = []
    with open(filename) as file:
        for line in file:
            layer_w = []
            for substring in line.split(';'):
                neuron_w = tuple((float(s) for s in substring.split(',')))
                layer_w.append(neuron_w)
            w.append(layer_w)
    return w


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Neuron:
    def __init__(self, weights):
        self.bias = weights[0]
        self.weights = weights[1:]

