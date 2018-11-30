import sys
import random

RANDOM = random.Random(123)

PURPLE = '\033[95m'
CYAN = '\033[96m'
DARKCYAN = '\033[36m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
END = '\033[0m'


USAGE = '''
{normal}Como usar:
    {code}python3 configure_network.py <nome> <lambda> <camada_1> <camada_2> ... <camada_n>{end}

{normal}Exemplo:
    {code}python3 configure_network.py pima 0.3 10 4 2 3{end}

{normal}Isso vai gerar dois arquivos. Um deles chamado "pima_rede.txt" e outro
chamado "pima_pesos.txt".

O arquivo de redes vai conter a inforação do parâmetro de regularização, bem
como o número de neurônios por camada.

O arquivo de pesos terá os pesos iniciais dessa rede.

Basta então executar:
    {code}python3 main.py pima_rede.txt pima_pesos.txt datasets/pima.txt{end}
'''.format(bold=BOLD, end=END, normal=GREEN, code=YELLOW+BOLD)


def random_weights(nodes_per_layer):
    weights = []
    for x, y in zip(nodes_per_layer[:-1], nodes_per_layer[1:]):
        # (x + 1) para representar o bias do neurônio
        weights.append([tuple(RANDOM.randint(1, 100) / 100
                              for _ in range(x + 1)) for _ in range(y)])
    return weights


if (len(sys.argv) < 3):
    print(USAGE)
    exit(1)

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
