import argparse
import data
import solve
import numpy as np

parser = argparse.ArgumentParser(description='It is a program for ML HW#1.')
parser.add_argument('file_path', help='file path of input', type=str)
parser.add_argument('base', help='the number of polynomial bases', type=int)
parser.add_argument('rate', help='rate of regulation', type=float)
args = parser.parse_args()

print('file_path: {}'.format(args.file_path))
print('base: {}'.format(args.base))
print('rate: {}'.format(args.rate))

(x, y) = data.readfile(args.file_path)
assert len(x) == len(y)
print(x)
# print(y)

features = solve.genfeatures(x, args.base)
matrix = solve.genmatrix(features)
b = solve.genb(features, y)
print(features)
print(matrix)
print(b)

answer = np.linalg.solve(matrix, b)
print(answer)