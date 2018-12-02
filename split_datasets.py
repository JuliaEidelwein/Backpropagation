import test_and_training as tt
import neural_net as nn


def to_string(instance):
    'Converte uma instância em uma linha de texto.'
    data = ','.join(map(str, instance.data))
    result = ','.join(map(str, instance.result))
    return data + ';' + result


datasets = ('ionosphere', 'wdbc', 'wine')

for filename in datasets:
    dataset = nn.parse_instances('datasets/' + filename + '.txt')
    folds = tt.stratified_k_fold(dataset)

    for size in range(1, 10):
        new_filename = '{0}_{1}.txt'.format(filename, size * 10)

        with open(new_filename, 'w') as file:
            file.write('\n') # Linha em branco no lugar do cabeçalho
            for i in range(0, size):
                fold = folds[i]
                for instance in fold:
                    file.write(to_string(instance) + '\n')


