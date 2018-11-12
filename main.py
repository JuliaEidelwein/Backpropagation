# encoding: utf-8
import sys
import numpy as np
import neural_net as nn
from collections import namedtuple



# Representa as instâncias lidas do dataset.
# "data" é a lista de valores de todos os atributos.
# "result" é a lista de valores experados na saída da rede.
Instance = namedtuple('Instance', ('data', 'result'))


def network_config(filename):
    '''
    Dado um nome de arquivo <filename> que representa a configuração da rede,
    retorna o parâmetro de regularização (lambda), e uma lista contendo o
    número de neurônios em cada camada.
    '''
    with open(filename) as file:
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
    '''
    initial_weights = []
    with open(filename) as file:
        for line in file:
            layer_w = []
            for substring in line.split(';'):
                neuron_w = tuple((float(s) for s in substring.split(',')))
                layer_w.append(neuron_w)
            initial_weights.append(layer_w)
    return initial_weights


def parse_instances(filename):
    '''
    Lê um arquivo e carrega as instâncias de treinamento definidas nele.
    Retorna uma lista contendo essas isntâncias (que são tuplas).
    '''
    dataset = []
    with open(filename) as file:
        next(file)  # Pula a linha com cabeçalhos.
        for line in file:
            data_string, result_string = line.split(';')
            data = tuple((float(s) for s in data_string.split(',')))
            result = [float(s) for s in result_string.split(',')]
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



# def normalize_dataset(D):
#     maxV = []
#     minV = []
#     for attr in range(len(D[0][0])):
#         maxV.append(max(D[i][0][attr] for i in range(len(D))))
#         minV.append(min(D[i][0][attr] for i in range(len(D))))
#     normalizedDataset = []
#     for instance in D:
#         normalizedAttr = []
#         for attr in range(len(instance[0])):
#             normalizedAttr.append(
#                 (((instance[0][attr] - minV[attr])*2)/(maxV[attr] - minV[attr])) - 1)
#         normalizedDataset.append((normalizedAttr, instance[1]))
#     print(maxV)
#     print(minV)
#     return normalizedDataset, minV, maxV


if __name__ == '__main__':

    if len(sys.argv) < 4:
        error_message = 'Modo de usar: \n\tpython3 main.py network.txt initial_weights.txt dataset.txt'
        print(error_message, file=sys.stderr)
        sys.exit(1)

    network_filename = sys.argv[1]
    initial_weights_filename = sys.argv[2]
    dataset_filename = sys.argv[3]

    reg_param, nodes_per_layer = network_config(network_filename)
    weights = network_weights(initial_weights_filename)
    dataset = parse_instances(dataset_filename)
    normalized_dataset, max_values, min_values = normalize_dataset(dataset)

    print('>> Pesos iniciais lidos: ', weights, end="\n\n")
    print('>> Dados lidos', dataset, end="\n\n")
    print('>> Dados normalizados', normalized_dataset, end="\n\n")

    layer_weights = nn.initialize_weights_as_matrix(weights, nodes_per_layer)

    n = 0
    z, activations = nn.propagate(layer_weights, dataset[n + 0], nodes_per_layer)
    errors = nn.J(activations[-1], dataset[n + 0].result)
    print(f">> Saída predita para o exemplo {n+1}: ", activations[-1])
    print(f">> Valor experado para o exemplo {n+1}: ", dataset[n + 0].result)
    print(">> Erros obtidos:", errors)
    print(nn.sigmoid(np.asarray([1, 2, 3])))
