# encoding: utf-8
import sys
import neural_net as nn


def network_config(filename):
    with open(filename) as file:
        regularization_parameter = float(next(file))
        nodes_per_layer = []
        for line in file:
            nodes_per_layer.append(int(line))
        return (regularization_parameter, nodes_per_layer)


def network_weights(filename):
    initial_weights = []
    with open(filename) as file:
        for line in file:
            layer_w = []
            for substring in line.split(';'):
                neuron_w = tuple((float(s) for s in substring.split(',')))
                layer_w.append(neuron_w)
            initial_weights.append(layer_w)
    return initial_weights


if __name__ == '__main__':

    if len(sys.argv) < 4:
        error_message = 'Modo de usar: \n\tpython3 main.py network.txt initial_weights.txt dataset.txt'
        print(error_message, file=sys.stderr)
        sys.exit(1)

    network_filename = sys.argv[1]
    initial_weights_filename = sys.argv[2]
    dataset_filename = sys.argv[3]

    reg_param, nodes_per_layer = network_config(network_filename)
    initial_weights = network_weights(initial_weights_filename)
    print(nodes_per_layer, initial_weights)

    net = nn.Network(nodes_per_layer, initial_weights)
    print(net.layers)
