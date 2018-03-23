import argparse
import data
import numpy as np

parser = argparse.ArgumentParser(description='It is a program for ML HW#2.')
parser.add_argument('train_img_path', help='file path of train img', type=str)
parser.add_argument('train_lbl_path', help='file path of train lbl', type=str)
parser.add_argument('test_img_path', help='file path of test img', type=str)
parser.add_argument('test_lbl_path', help='file path of test lbl', type=str)
args = parser.parse_args()

print('train_img_path: {}'.format(args.train_img_path))
print('train_lbl_path: {}'.format(args.train_lbl_path))
print('test_img_path: {}'.format(args.test_img_path))
print('test_img_path: {}'.format(args.test_lbl_path))