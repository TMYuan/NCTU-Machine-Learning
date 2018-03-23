import random

def read(file_path):
    """
    This function is used to read file.
    Convert file format into my data structure.
    Each element in x list is a line in the file.
    """
    x = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # print(line)
            if line != '':
                x.append([int(n) for n in list(line)])
    return x

def gen(file_path, number):
    """
    This function is used to generate fake file.
    """
    with open(file_path, 'w') as file:
        for _ in range(number):
            for __ in range(random.randint(0, 50)):
                file.write('{}'.format(random.randint(0,1)))
            file.write('\n')

if __name__ == '__main__':
    # gen('test.txt', 10)
    x = read('test.txt')
    print(x)