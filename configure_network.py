import sys

if (len(sys.argv) < 3):
    print("Erro")

name, ret_param, *nodes_per_layer = sys.argv[1:]

with(open(name + "_rede.txt", "w")) as file:
    file.write(ret_param)
    for n in nodes_per_layer:
        file.write('\n')
        file.write(n)
