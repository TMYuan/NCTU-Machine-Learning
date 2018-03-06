import random

def readfile(file_path):
    """
    This function is used to read file.
    Convert file format into my data structure.
    list x store data point, list y store target.
    """
    x = []
    y = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split(',')
            # print(line)
            if line[0] != '' and line[1] != '':
                x.append(line[0])
                y.append(line[1])
    return (x, y)
