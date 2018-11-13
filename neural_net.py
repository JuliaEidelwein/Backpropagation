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
    bias_weight = np.array([1])
    current_activation = np.asarray(instance.data)
    for w_i in weights:
        value = np.dot(w_i, np.concatenate((bias_weight, current_activation)))
        z.append(value)
        current_activation = sigmoid(value).A1
        activations.append(current_activation)
    return z, activations


def calculate_error(y, fx):
    '''
    Calcula o erro entre o valor esperado "y" e o valor obtido "f(x)".
    '''
    return -y * ln(fx) - (1 - y) * ln(1 - fx)


def J(outputs, expected_outputs):
    '''
    Recebe uma lista de resultados e uma lista de valores esperados.
    Retorna uma lista com os valores dos erros das entradas.
    '''
    return [calculate_error(y, fx) for y, fx in zip(expected_outputs, outputs)]
