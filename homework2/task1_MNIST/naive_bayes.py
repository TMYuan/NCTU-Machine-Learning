import numpy as np
import pandas as pd
import math
import os.path

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

def gaussian(x, mean, sigma):
    # return math.log((1 / (math.sqrt(2 * math.pi()) * sigma)) * (math.exp(-0.5 * ((x - mean)/sigma) ** 2)))
    # if sigma <= 0.1:
    #     if x == mean:
    #         x = 0
    #     else:
    #         x = -1000
    # else:
    if sigma < 10:
        sigma = 10
    answer = math.log(1 / (sigma * math.sqrt(2 * math.pi))) - (0.5 * (math.pow((x-mean) / sigma, 2)))
    return answer

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
    if os.path.isfile('./lookup.npy'):
        look_up = np.load('./lookup.npy')
    else:
        for i in range(look_up.shape[0]):
            for j in range(look_up.shape[1]):
                class_count = len(train_df[train_df['target'] == i])
                for k in range(look_up.shape[2]):
                    # Laplacian Smoothing
                    count = len(train_df[(train_df[j] == k) & (train_df['target'] == i)])
                    look_up[i][j][k] = math.log((count + 1) / (class_count + 32*1))
        np.save('./lookup.npy', look_up)

    # Calculate each feature probability for each class in each row in test case
    prediction = []
    for i in range(test_img.shape[0]):
        print('-'*20)
        print('No. {} data'.format(i))
        posterier = prior.copy()
        for j in range(10):
            for k in range(test_img.shape[1]):
                posterier[j] += look_up[j][k][test_img[i][k]]
        prediction.append(np.argmax(posterier))
        print('Posterior: {}'.format(posterier))
        print('Prediction: {}'.format(np.argmax(posterier)))
        print('Answer: {}'.format(test_lbl[i]))
    prediction = np.array(prediction)
    error = np.count_nonzero(test_lbl != prediction) / len(test_lbl)
    print('Error rate: {}'.format(error))

def continuousClassify(train_img, train_lbl, test_img, test_lbl):
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
    look_up = np.zeros((10, train_img.shape[1], 2))
    if os.path.isfile('./lookup_gau.npy'):
        look_up = np.load('./lookup_gau.npy')
    else:
        for i in range(10):
            target = train_df[train_df['target'] == i].drop(['target'], axis=1).as_matrix()
            look_up[i, :, 0] = np.mean(target, axis=0)
            look_up[i, :, 1] = np.std(target, axis=0)
        print(look_up[0, :, 0])
        print(look_up.shape)
        np.save('./lookup_gau.npy', look_up)


    # Calculate each feature probability for each class in each row in test case
    prediction = []
    for i in range(test_img.shape[0]):
        print('-'*20)
        print('No. {} data'.format(i))
        posterier = prior.copy()
        for j in range(10):
            for k in range(test_img.shape[1]):
                x = test_img[i][k]
                mean = look_up[j][k][0]
                std = look_up[j][k][1]
                posterier[j] += gaussian(x, mean, std)
        prediction.append(np.argmax(posterier))
        print('Posterior: {}'.format(posterier))
        print('Prediction: {}'.format(np.argmax(posterier)))
        print('Answer: {}'.format(test_lbl[i]))
    prediction = np.array(prediction)
    error = np.count_nonzero(test_lbl != prediction) / len(test_lbl)
    print('Error rate: {}'.format(error))


def classify(train_img, train_lbl, test_img, test_lbl, mode):
    if mode == 0:
        train_img, test_img = discreteData(train_img, test_img)
        discreteClassify(train_img, train_lbl, test_img, test_lbl)
    else:
        continuousClassify(train_img, train_lbl, test_img, test_lbl)