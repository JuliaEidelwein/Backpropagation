from collections import namedtuple
import sys
import csv
import random
import test_and_training as tat


def read_data(filename):
    '''
    Reads a CSV file and returns the attibute's names and a list of register tuples
    '''
    data = []
    fields = None
    with open(filename) as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        fields = tuple(map(lambda s: s.lower(), next(reader)))
        Data = namedtuple('Data', fields)
        for csv_row in reader:
            data.append(Data(*csv_row))
    return (fields, data)


if __name__ == '__main__':

    if len(sys.argv) > 1:
        csv_filename = sys.argv[1]
    else:
        raise Exception('Provide a CSV file name.\nExample: python3 main.py data.csv')

    fieldnames, rows = read_data(csv_filename)

    numeric_fields = set()  # Set of fields with numeric value
    for name in sys.argv[2:]:
        numeric_fields.add(name)

    attributes = fieldnames[:-1]