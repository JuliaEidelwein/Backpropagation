import sys
import random

RANDOM = random.Random(123)


def random_weights(nodes_per_layer):
    weights = []
    for x, y in zip(nodes_per_layer[:-1], nodes_per_layer[1:]):
        # (x + 1) para representar o bias do neurônio
        weights.append([tuple(RANDOM.random()
                              for _ in range(x + 1)) for _ in range(y)])
    return weights


if (len(sys.argv) < 3):
    print("Erro")

name, ret_param, *nodes_per_layer = sys.argv[1:]

with(open(name + "_rede.txt", "w")) as file:
    file.write(ret_param)
    for n in nodes_per_layer:
        file.write('\n')
        file.write(n)

with(open(name + "_pesos.txt", 'w')) as file:
    weights = random_weights(list(map(int, nodes_per_layer)))
    for line in weights:
        line_string = ';'.join([','.join(map(str, w)) for w in line])
        file.write(line_string + '\n')
