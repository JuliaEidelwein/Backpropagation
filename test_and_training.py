import random
import math
from collections import namedtuple, defaultdict
from statistics import mean, stdev
import neural_net as nn

Bootstrap = namedtuple('Bootsrap', ('training', 'test'))

RANDOM = random.Random(123)


def bootstrap(data, r=10):
    """
    Splits data D in r test and training sets with resampling
    """
    bootstrap_sets = []

    # Creates r bootstraps
    for _ in range(r):
        training_set = []

        # Selects n random instances (with resampling) for the training set
        for _ in range(len(data)):
            training_set.append(random.choice(data))
        test_set = [inst for inst in data if inst not in training_set]
        bootstrap_sets.append(Bootstrap(training=training_set, test=test_set))
    return bootstrap_sets


def stratified_k_fold(data, k=10):
    '''
    Splits data in k partitions, maintaining instances per class proportion in each fold
    '''
    instances_per_class = defaultdict(list)

    for instance in data:
        instances_per_class[instance.klass].append(instance)
    amount_per_class = defaultdict(list)

    for c in instances_per_class:
        # Stores a integer number of instances per fold and the remainder of division by k
        amount_per_class[c] = [
            len(instances_per_class[c]) // k, len(instances_per_class[c]) % k]
    folds = [[] for l in range(k)]

    for f in range(k):
        for c in instances_per_class:
            # Distributes an equal amount of instances to each fold
            for _ in range(amount_per_class[c][0]):
                folds[f].append(instances_per_class[c].pop())
            # And one spare instance to the first remainder of division by k folds
            if amount_per_class[c][1] >= 1:
                folds[f].append(instances_per_class[c].pop())
                amount_per_class[c][1] = amount_per_class[c][1] - 1
    return folds


def nested_dict(n):
    if n is 1:
        return defaultdict(int)
    else:
        return defaultdict(lambda: nested_dict(n - 1))


def sum_tp_fp_fn(confusion_matrix, target_class=None):
    '''
    Returns the sum of true positives, false positives and false negatives
    target_class = class to have its sum returned: if None, sums all classes
    '''
    tp = 0
    fp = 0
    fn = 0
    if target_class is not None:
        tp = confusion_matrix[target_class][target_class]
        for class_name in confusion_matrix:
            if target_class != class_name:
                fp += confusion_matrix[class_name][target_class]
                fn += confusion_matrix[target_class][class_name]
    else:
        for c1 in confusion_matrix:
            for c2 in confusion_matrix:
                if c1 != c2:
                    fp += confusion_matrix[c2][c1]
                    fn += confusion_matrix[c1][c2]
                else:
                    tp += confusion_matrix[c1][c2]
    return tp, fp, fn


def precision(tp, fp):
    return tp / (tp + fp)


def recall(tp, fn):
    return tp / (tp + fn)


def f_measure(beta, tp, fp, fn):
    try:
        b2 = beta**2
        prec = precision(tp, fp)
        rec = recall(tp, fn)
        return (1 + b2) * ((prec * rec) / (b2 * prec + rec))

    except ZeroDivisionError:
        return 0


def get_classes(instances):
    classes = {instance.klass for instance in instances}
    return list(classes)


def cross_validation(weights, reg_param, instances, k=10, r=10):
    folds = stratified_k_fold(instances, k)
    classes = get_classes(instances)
    n_classes = len(classes)
    confusion_matrix = nested_dict(n_classes)
    fmeasures = []
    Jmean = 0
    for current_fold in folds:
        for c1 in classes:
            for c2 in classes:
                confusion_matrix[c1][c2] = 0
        training_data = []
        for fold in folds:
            if fold != current_fold:
                for instance in fold:
                    training_data.append(instance)

        net = nn.Network(weights, reg_param)
        # TODO: Resolver detalhes do treinamento
        print("Treinando {n} instâncias".format(n=len(training_data)))
        J = net.train(training_data)
        Jmean = Jmean + J
        print("J final: " + str(J))

        for instance in current_fold:
            prediction = net.predict_class(instance)
            print('Decisão: ', prediction)
            confusion_matrix[instance.klass][prediction] += 1

        # TODO: Verificar se está certo isto.
        tp, fp, fn = sum_tp_fp_fn(confusion_matrix)
        fmeasures.append(f_measure(1, tp, fp, fn))
    Jmean = Jmean/k
    print("J médio é: " + str(Jmean))
    # Retorna a média e o desvio padrão das fmeasures
    return (mean(fmeasures), stdev(fmeasures))
