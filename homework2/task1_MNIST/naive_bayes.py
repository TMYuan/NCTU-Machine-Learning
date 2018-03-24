import numpy as np
import pandas as pd
import math

def discreteData(train_img, test_img, bins=32):
    interval = 256 // bins
    train_img = train_img // interval
    test_img = test_img // interval
    return (train_img, test_img)

def calculateGaussian(train_img):
    num_feature = train_img.shape[1]
    gaussian_map = np.zeros((num_feature, 2))
    gaussian_map[:, 0] = np.mean(train_img, axis=0)
    gaussian_map[:, 1] = np.var(train_img, axis=0)
    return gaussian_map

def discreteClassify(train_img, train_lbl, test_img, test_lbl):
    # Calculate P(Y)(prior), in this case, P(0), P(1), etc
    prior = np.zeros(10)
    for i in range(10):
        total = len(train_lbl)
        prior[i] = math.log(np.count_nonzero(train_lbl == i) / total)
    print(prior)

    # Generate dataframe from training data.
    train_df = pd.DataFrame(train_img)
    train_df['target'] = train_lbl
    print(train_df)

    # Prepare a matrix to store every combination for use
    look_up = np.zeros((10, train_img.shape[1], 32))
    for i in range(look_up.shape[0]):
        for j in range(look_up.shape[1]):
            class_count = len(train_df['target'] == i)
            for k in range(look_up.shape[2]):
                count = len(train_df[(train_df[j] == k) & (train_df['target'] == i)])
                look_up[i][j][k] = math.log((count + 1) / (class_count + 32*1))

    # Calculate each feature probability for each class in each row in test case
    for i in range(test_img.shape[0]):
        print('-'*20)
        print('No. {} data'.format(i))
        posterier = np.zeros(10)
        for j in range(10):
            posterier[j] += prior[j]
            for k in range(test_img.shape[1]):
                posterier[j] += look_up[j][k][test_img[i][k]]
        print(posterier)


def classify(train_img, train_lbl, test_img, test_lbl, mode):
    if mode == 0:
        train_img, test_img = discreteData(train_img, test_img)
        discreteClassify(train_img, train_lbl, test_img, test_lbl)
    else:
        gaussian_map = calculateGaussian(train_img)
        print(gaussian_map.shape)