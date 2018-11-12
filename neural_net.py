import numpy as np
import math


def sigmoid(fx):
    return 1 / (1 + np.exp(-fx))


def create_network(nodes_per_layer, weights):
    network = []
    for l in range(1, len(nodes_per_layer)):
        layer = []
        for n in range(nodes_per_layer[l]):
            layer.append(Neuron(weights[l-1][n]))
        network.append(layer)
    return network


def initialize_weights_as_matrix(weights, nodes_per_layer):
    layer_weights = []
    for l in range(1, len(nodes_per_layer)):
        layer_weights.append(np.matrix(weights[l-1]))
    return layer_weights


def propagate(layer_weights, input, nodes_per_layer):
    z = []
    activations = []
    current_activation = np.asarray(input[0])
    # print(current_activation)
    for l in range(1, len(nodes_per_layer)):
        z.append(np.dot(
            layer_weights[l-1],
            np.concatenate((np.array([1]), current_activation), axis=None)))
        current_activation = sigmoid(z[l-1]).A1
        activations.append(current_activation)
    # print(f'>> Z: {z}\n>> Activations: {activations}')
    return z, activations


def calculate_error(y, fx):
    '''
    Calcula o erro entre o valor esperado "y" e o valor obtido "f(x)".
    '''
    return -y * math.log(fx) - (1 - y) * (math.log(1 - fx))


def J(outputs, expected_outputs):
    '''
    Recebe uma lista de resultados e uma lista de valores esperados.
    Retorna uma lista com os valores dos erros das entradas.
    '''
    return [calculate_error(y, fx) for y, fx in zip(expected_outputs, outputs)]


class Neuron:
    def __init__(self, weights=None, activation=None):
        if(weights):
            self.bias = weights[0]
            self.weights = weights[1:]
        else:
            self.bias = None
            self.weights = None
        self.activation = activation

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
