import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Neuron:
    def __init__(self, weights):
        self.bias = weights[0]
        self.weights = weights[1:]

    def dot_product(self, values):
        return np.dot(values, self.weights) + self.bias

    def __repr__(self):
        return f'Neuron(bias={self.bias}, weights={self.weights}'


class Network:
    def __init__(self, nodes_per_layer, initial_weights):
        self.initial_layer = nodes_per_layer.pop(0)
        self.layers = []
        for weights in initial_weights:
            layer = []
            for weight in weights:
                layer.append(Neuron(weight))
            self.layers.append(layer)
