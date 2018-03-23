import argparse
import numpy as np

parser = argparse.ArgumentParser(description='It is a program for ML HW#1.')
parser.add_argument('file_path', help='file path of input', type=str)
parser.add_argument('beta_a', help='value "a" of initial beta function', type=int)
parser.add_argument('beta_b', help='value "b" of initial beta function', type=int)
args = parser.parse_args()

print('file_path: {}'.format(args.file_path))
print('beta_a: {}'.format(args.base))
print('beta_b: {}'.format(args.rate))