import argparse
import data
import numpy as np

parser = argparse.ArgumentParser(description='It is a program for ML HW#1.')
parser.add_argument('file_path', help='file path of input', type=str)
parser.add_argument('beta_a', help='value "a" of initial beta function', type=int)
parser.add_argument('beta_b', help='value "b" of initial beta function', type=int)
args = parser.parse_args()

print('file_path: {}'.format(args.file_path))
print('beta_a: {}'.format(args.beta_a))
print('beta_b: {}'.format(args.beta_b))

# read data from the path
x = data.read(args.file_path)

# for each element of x, calculate the count of 0 and the count of 1
beta_a = args.beta_a
beta_b = args.beta_b
for batch in x:
    count_zero = batch.count(0)
    count_one = batch.count(1)
    print('-'*20)
    print('Prior: Beta({}, {})'.format(beta_a, beta_b))
    print('Binmail Likelihood: {}'.format((beta_a+count_one)/(beta_a+count_one+beta_b+count_zero)))
    beta_a = beta_a + count_one
    beta_b = beta_b + count_zero
    print('Posterior: Beta({}, {})'.format(beta_a, beta_b))
