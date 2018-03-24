import argparse
import mnist
import naive_bayes as nb
import numpy as np

parser = argparse.ArgumentParser(description='It is a program for ML HW#2.')
parser.add_argument('train_img_path', help='file path of train img', type=str)
parser.add_argument('train_lbl_path', help='file path of train lbl', type=str)
parser.add_argument('test_img_path', help='file path of test img', type=str)
parser.add_argument('test_lbl_path', help='file path of test lbl', type=str)
parser.add_argument('mode', help='toggle option', type=int)
args = parser.parse_args()
print('train_img_path: {}'.format(args.train_img_path))
print('train_lbl_path: {}'.format(args.train_lbl_path))
print('test_img_path: {}'.format(args.test_img_path))
print('test_img_path: {}'.format(args.test_lbl_path))

train_img, train_lbl = mnist.read(args.train_img_path, args.train_lbl_path)
test_img, test_lbl = mnist.read(args.test_img_path, args.test_lbl_path)
print(train_img.shape)

nb.classify(train_img, train_lbl, test_img, test_lbl, args.mode)