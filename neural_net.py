from math import log as ln
from collections import namedtuple
import numpy as np


# Representa as instâncias lidas do dataset.
# "data" é a lista de valores de todos os atributos.
# "result" é a lista de valores experados na saída da rede.
class Instance(namedtuple('BasicInstance', ('data', 'result'))):
    def __repr__(self):
        data = ', '.join(map(str, self.data))
        result = ', '.join(map(str, self.result))
        return f'{{ data: ({data}), result: ({result}) }}'


def network_config(filename):
    '''
    Dado um nome de arquivo <filename> que representa a configuração da rede,
    retorna o parâmetro de regularização (lambda), e uma lista contendo o
    número de neurônios em cada camada.
    '''
    with open(filename, 'r') as file:
        regularization_parameter = float(next(file))
        nodes_per_layer = []
        for line in file:
            nodes_per_layer.append(int(line))
    return (regularization_parameter, nodes_per_layer)


def network_weights(filename):
    '''
    Dado um nome de arquivo <filename> que contém os pesos inicias de cada
    neurônio, retorna uma lista de tuplas, que contém os pesos iniciais desses
    neurônios.

    Retorna uma lista contendo matrizes de pesos.  Cada matriz corresponde
    aos pesos de umas das camadas da rede.
    '''
    initial_weights = []
    with open(filename, 'r') as file:
        for line in file:
            layer_w = []
            for substring in line.split(';'):
                neuron_w = tuple((float(s) for s in substring.split(',')))
                layer_w.append(neuron_w)
            initial_weights.append(layer_w)
    return [np.matrix(w) for w in initial_weights]


def parse_instances(filename):
    '''
    Lê um arquivo e carrega as instâncias de treinamento definidas nele.
    Retorna uma lista contendo essas isntâncias (que são tuplas).
    '''
    dataset = []
    with open(filename, 'r') as file:
        next(file)  # Pula a linha com cabeçalhos.
        for line in file:
            data, result = [tuple(map(float, part.split(',')))
                            for part in line.split(';')]
            dataset.append(Instance(data=data, result=result))
    return dataset


def normalize_dataset(dataset):
    '''
    Normalisa cada atributo de cada instância no dataset e retorna uma nova
    lista de instâncias com valores normalizados entre -1 e 1.
    '''
    max_vs = []
    min_vs = []
    for index in range(len(dataset[0].data)):
        values = tuple(instance.data[index] for instance in dataset)
        max_vs.append(max(values))
        min_vs.append(min(values))
    normalized_dataset = []
    for instance in dataset:
        data = tuple(2 * ((x - min_vs[index]) / (max_vs[index] - min_vs[index])) - 1
                     for index, x in enumerate(instance.data))
        normalized_dataset.append(Instance(data=data, result=instance.result))
    return (normalized_dataset, max_vs, min_vs)


def sigmoid(fx):
    return 1 / (1 + np.exp(-fx))


def propagate(weights, instance):
    z = []
    activations = []
    bias_value = np.array([1])
    current_activation = np.asarray(instance.data)
    for w_i in weights:
        value = np.dot(w_i, np.concatenate((bias_value, current_activation)))
        z.append(value)
        current_activation = sigmoid(value).A1
        activations.append(current_activation)
    return z, activations


def calculate_cost(y, fx):
    '''
    Calcula o erro entre o valor esperado "y" e o valor obtido "f(x)".
    '''
    return -y * ln(fx) - (1 - y) * ln(1 - fx)


def J(outputs, expected_outputs):
    '''
    Recebe uma lista de resultados e uma lista de valores esperados.
    Retorna uma lista com os valores dos erros das entradas.
    '''
    return [calculate_cost(y, fx) for y, fx in zip(expected_outputs, outputs)]


def outputDelta(outputs, expected_outputs):
    return [out - expected for expected,out in zip(expected_outputs, outputs)]

def innerDelta(deltas, weights, activations):
    deltaSum = np.transpose(weights)*np.transpose(np.asmatrix(deltas))
    activations = np.concatenate((np.array([1]), activations), axis=None)
    deltaSum = [x * y for x, y in zip(deltaSum, activations)]
    deltaSum = [x * y for x, y in zip(deltaSum, (1-activations))]
    return deltaSum

def backpropagation(layer_weights, inputs, nodes_per_layer, reg_param, alpha):
    D = []
    for l in layer_weights:
        D.append(np.asmatrix([np.zeros(n.size) for n in l]))
    # D = [np.zeros(y.size) for x,y in layer_weights]
    for input in inputs:
        z, activations = propagate(layer_weights,input)
        outerDeltas = outputDelta(activations[-1],np.asarray(input[-1]))
        deltas = []
        deltas.append(outerDeltas)
        layer_deltas = outerDeltas
        numOfLayers = len(nodes_per_layer)
        for layer in range(numOfLayers-2,0,-1):
            if(layer == 0):
                layer_deltas = innerDelta(layer_deltas,layer_weights[layer],input[0])
            else:
                layer_deltas = innerDelta(layer_deltas,layer_weights[layer],activations[layer-1])
            layer_deltas.pop(0)
            layer_deltas = [x.item(0) for x in layer_deltas]
            deltas.append(layer_deltas)
        inputC = np.concatenate((np.array([1]), input[0]), axis=None)
        for layer in range(numOfLayers - 2, -1, -1):
            activations[layer-1] = np.concatenate((np.array([1]), activations[layer-1]), axis=None)
            if(layer==0):
                D[layer] = D[layer] + np.transpose(np.asmatrix(deltas[numOfLayers-2]))*np.asmatrix(inputC)
            else:
                D[layer] = D[layer] + np.transpose(np.asmatrix(deltas[numOfLayers - 2 - layer]))*np.asmatrix(activations[layer - 1])
    n = len(inputs)
    for layer in range(numOfLayers - 2, -1, -1):
        P = reg_param*layer_weights[layer]
        D[layer] = (1/n)*(D[layer]+P)
    for layer in range(numOfLayers - 2, -1, -1):
        layer_weights[layer] = layer_weights[layer] - alpha*D[layer]



def numerical_gradient_estimation(epsilon, inputs, layer_weights):
    for layer in range(len(layer_weights)):
        for weight in range(len(layer)):
            layer_weights_temp = layer_weights
            layer_weights_temp[layer][weight] = layer_weights_temp[layer][weight] + epsilon
            propagate(layer_weights_temp,)
