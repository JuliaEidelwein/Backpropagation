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
    return 1 / (1 + np.exp(-x))

def create_network(nodes_per_layer, weights):
    network = []
    for l in range(1, len(nodes_per_layer)):
        layer = []
        for n in range(nodes_per_layer[l]):
            layer.append(Neuron(weights[l-1][n]))
        network.append(layer)
    return network

def initialize_weights_as_matrix(weights, nodes_per_layer):
    layerWeights = []
    for l in range(1, len(nodes_per_layer)):
        layerWeights.append(np.matrix(weights[l-1]))
    return layerWeights


# def propagate(instance, network, input):
#     for layer in network:
#         for neuron in layer:
#             neuron.activation =

def propagate(layerWeights, input, nodes_per_layer):
    z = []
    activations = []
    currentActivation = np.asarray(input[0])
    print(currentActivation)
    for l in range(1, len(nodes_per_layer)):
        z.append(np.dot(layerWeights[l-1],np.concatenate((np.array([1]),currentActivation),axis=None)))
        currentActivation = sigmoid(z[l-1]).A1
        activations.append(currentActivation)
    print("Z:")
    print(z)
    print(activations)
    return z, activations

def J(outputs, expectedOutputs):
    sum = 0
    print(outputs)
    print(expectedOutputs)
    for i in range(len(expectedOutputs)):
        # sum = sum + (-expectedOutputs[i])*math.log10(outputs[i]) \
        #       -(1-expectedOutputs[i])*math.log10(1 - outputs[i])
        part1 = (-expectedOutputs[i])*math.log10(outputs[i])
        part2 = (1-expectedOutputs[i])*math.log10(1 - outputs[i])
        print(part1)
        print(part2)
        print(part1-part2)
    sum = (-(0.9)*math.log2(0.79403))-((1-0.9)*math.log2(1 - 0.79403))
    return sum
    # return sum/len(expectedOutputs)

class Neuron:
    def __init__(self, weights = None, activation = None):
        if(weights):
            self.bias = weights[0]
            self.weights = weights[1:]
        else:
            self.bias = None
            self.weights = None
        self.activation = activation

