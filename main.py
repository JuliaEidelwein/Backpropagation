# encoding: utf-8

# from collections import namedtuple
# import sys
# import csv
# import random
# import test_and_training as tat

import sys
import neural_net as nn

if __name__ == '__main__':

    if len(sys.argv) < 4:
        print('Número de argumentos insuficiente', file=sys.stderr)
        sys.exit(1)

    network_filename = sys.argv[1]
    initial_weights_filename = sys.argv[2]
    dataset_filename = sys.argv[3]

    reg_param, nodes_per_layer = nn.network_config(network_filename)
    weights = nn.network_weights(initial_weights_filename)
    print(weights)

