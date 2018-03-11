import argparse
import data
import solve
import newton
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
print('-'*30)
print('x: {}'.format(x))
# print(y)


weight_LSE, error_LSE = solve.LSE(x, y, args.base, args.rate)
print('-'*30)
print('weight of LSE: \n{}'.format(weight_LSE))
print('error of LSE: \n{}'.format(error_LSE))

weight_NT, error_NT = newton.optimize(x, y, args.base)
print('-'*30)
print('weight of NT: \n{}'.format(weight_NT))
print('error of NT: \n{}'.format(error_NT))

# print(np.array_equal(weight_LSE[0][0], weight_NT[0][0]))
# print(weight_LSE.dtype)
# print(weight_NT.dtype)