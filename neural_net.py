from math import log as ln
from collections import namedtuple
import numpy as np


# Representa as instâncias lidas do dataset.
# "data" é a lista de valores de todos os atributos.
# "result" é a lista de valores esperados na saída da rede.
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
    weights = []
    with open(filename, 'r') as file:
        for line in file:
            layer_w = []
            for substring in line.split(';'):
                neuron_w = tuple((float(s) for s in substring.split(',')))
                layer_w.append(neuron_w)
            weights.append(layer_w)
    return weights


def parse_instances(filename):
    '''
    Lê um arquivo e carrega as instâncias de treinamento definidas nele.
    Retorna uma lista contendo essas instâncias (que são tuplas).
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
    Normaliza cada atributo de cada instância no dataset e retorna uma nova
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


class Network:
    def __init__(self, weights, reg_param):
        # TODO: Não sei onde está definido o alpha
        self.alpha = 0.9
        self.reg_param = reg_param
        self.layers = [np.array(w) for w in weights]
        self.activations = []
        self.gradients = None
        self.deltas = []

    def sigmoid(self, fx):
        return 1 / (1 + np.exp(-fx))

    def sigmoid_derivative(self, s):
        return s * (1 - s)

    def train(self, instances):
        n = len(instances)
        for instance in instances:
            z, activations = self.activate(instance.data)
            output = activations[-1][1:]
            jCost = self.cost(instance[1],output)
            deltas = self.calculate_deltas(instance.result, activations)
            self.update_gradients(deltas, activations)

            for i, d in enumerate(deltas):
                print(f'Delta {i}')
                print(d)
            for i, a in enumerate(activations):
                print(f'Activation {i}')
                print(a[1:])
            for i, w in enumerate(self.layers):
                print(f'Weights {i}')
                print(w)

            for i, g in enumerate(self.gradients):
                print(f'Gradiente {i}')
                print(g)
        gradients = self.calculate_regularized_gradients(n)
        for i, g in enumerate(gradients):
            print(f'Gradiente {i}')
            print(g)
        print("J!!!!!!!!!!!!")
        print(self.J(instances))
        self.update_weights()
        print(self.J(instances))


    def activate(self, values):
        zs = []
        self.activations = [np.array((1,) + values)]
        for thetas in self.layers:
            # Insere 1 na matriz
            values = np.insert(values, 0, 1, axis=0)
            z = thetas.dot(values.T)
            zs.append(z)
            a = self.sigmoid(z)
            values = a
            a = np.insert(a, 0, 1, axis=0)
            self.activations.append(a)
        return (zs, self.activations)

    def calculate_deltas(self, expected, activations):
        fx = activations[-1][1:]
        d = fx - expected
        self.deltas = [d]
        n = len(self.layers) - 1
        for w, a in zip(reversed(self.layers[1:]), reversed(activations[1:-1])):
            d = w.T.dot(d) * self.sigmoid_derivative(a)
            d = d[1:] # Remove o delta associado ao viés da camada
            self.deltas.append(d)
        self.deltas.reverse()
        return self.deltas

    def update_gradients(self, deltas, activations):
        if self.gradients is None:
            self.gradients = []
            for i in range(len(self.layers)):
                self.gradients.append(np.asmatrix(np.zeros((len(self.deltas[i]),len(self.activations[i])))))
        numOfLayers = len(self.layers)
        for layer in range(numOfLayers - 1, -1, -1):
            if(layer==0):
                print(np.transpose(np.asmatrix(deltas[0]))*np.asmatrix(activations[0]))
                self.gradients[layer] = self.gradients[layer] + np.transpose(np.asmatrix(deltas[0]))*np.asmatrix(activations[0])
            else:
                print(np.transpose(np.asmatrix(deltas[layer]))*np.asmatrix(activations[layer+1]))
                self.gradients[layer] = self.gradients[layer] + np.transpose(np.asmatrix(deltas[layer]))*(np.asmatrix(activations[layer]))

    def calculate_regularized_gradients(self, n):
        print(f'Lambda: {self.reg_param}')
        for k in range(len(self.layers)):
            pk = self.reg_param * self.layers[k]
            for l in pk:
                l[0] = 0
            g = (1/n) * (self.gradients[k] + pk)
            self.gradients[k] = g
        return self.gradients

    def update_weights(self):
        for i in range(len(self.layers)):
            self.layers[i] = np.asarray(self.layers[i] - (self.alpha * self.gradients[i]))

    def _cost(self, y, fx):
        return -y * ln(fx) - (1 - y) * ln(1 - fx)

    def cost(self, expected_output, output):
        '''
        Calcula o J da ativação
        '''
        return sum(self._cost(y, fx) for y, fx in zip(expected_output, output))

    def J(self, instances):
        j = 0
        nInstances = len(instances)
        for instance in instances:
            z, activations = self.activate(instance.data)
            output = activations[-1][1:]
            j = j + self.cost(instance[1],output)
        j = j/nInstances
        S = self.calculateS(nInstances)
        return j+S


    def calculateS(self, n):
        S = 0
        for layer in self.layers:
            for ne in layer:
                for w in range(len(ne)):
                    if w == 0:
                        S = S
                    else:
                        S = S + ne[w]*ne[w]
        S = (self.reg_param/(2*n))*S
        return S
