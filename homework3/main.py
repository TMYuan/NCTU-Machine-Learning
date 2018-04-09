import data
import numpy as np
import random

def _estimator(mean_old, SSDM_old, variance_old, point, count):
    mean = mean_old + (point - mean_old)/count
    SSDM = SSDM_old + (point - mean_old) * (point - mean)
    variance = SSDM / (count - 1)
    return (mean, SSDM, variance, count + 1)

def estimator(mean, variance):
    mean_est = data.gaussianGenerator(mean, variance)
    SSDM = 0
    variance_est = 0
    count = 2
    while(True):
        point = data.gaussianGenerator(mean, variance)
        (mean_est_new, SSDM, variance_est_new, count) = _estimator(mean_est, SSDM, variance_est, point, count)
        print('-'*20)
        print('Current Point: {}'.format(point))
        print('Estimate Mean: {}'.format(mean_est_new))
        print('Estimate variance: {}'.format(variance_est_new))
        if (abs(mean_est_new - mean_est) < 0.00001) & (abs(variance_est_new - variance_est) < 0.00001):
            break
        else:
            mean_est = mean_est_new
            variance_est = variance_est_new

def BLR(basis, a, b, weight):
    x = random.uniform(-10, 10)
    y = data.polyBasisModel(basis, a, weight, x)
    design_matrix = data.designMatrix(basis, x)
    prior_mean = 0
    prior_var_inv = b

    # For the first point
    posterier_var_inv = a * np.matmul(design_matrix.T, design_matrix) + b * np.eye(basis)
    posterier_mean = a * np.matmul(np.linalg.inv(posterier_var_inv), design_matrix.T) * y
    predictive_distribution_mean = np.matmul(design_matrix, posterier_mean)
    predictive_distribution_var = 1/a + \
        np.matmul(np.matmul(design_matrix, np.linalg.inv(posterier_var_inv)), design_matrix.T)
    
    print('-'*20)
    print('Current Data Point: ({}, {})'.format(x, y))
    print('Posterier Parameter')
    print('mean:\n{},\nco-variance: \n{}'.format(posterier_mean, np.linalg.inv(posterier_var_inv)))
    print('Predictive Distribution')
    print('mean:\n{},\nco-variance: \n{}'.format(predictive_distribution_mean, predictive_distribution_var))

    # Online Learning
    while(True):
        x = random.uniform(-10, 10)
        y = data.polyBasisModel(basis, a, weight, x)
        design_matrix = data.designMatrix(basis, x)
        prior_mean = posterier_mean.copy()
        prior_var_inv = posterier_var_inv.copy()

        posterier_var_inv = a * np.matmul(design_matrix.T, design_matrix) + prior_var_inv
        posterier_mean = np.matmul(np.linalg.inv(posterier_var_inv),\
            a * design_matrix.T * y + np.matmul(prior_var_inv, prior_mean))
        predictive_distribution_mean = np.matmul(design_matrix, posterier_mean)
        predictive_distribution_var = 1/a + \
            np.matmul(np.matmul(design_matrix, np.linalg.inv(posterier_var_inv)), design_matrix.T)
        print('-'*20)
        print('Current Data Point: ({}, {})'.format(x, y))
        print('Posterier Parameter')
        print('mean:\n{},\nco-variance: \n{}'.format(posterier_mean, np.linalg.inv(posterier_var_inv)))
        print('Predictive Distribution')
        print('mean:\n{},\nco-variance: \n{}'.format(predictive_distribution_mean, predictive_distribution_var))

        if (abs(np.sum(prior_mean - posterier_mean)) < 0.001) &\
            (abs(np.sum(prior_var_inv - posterier_var_inv)) < 0.001):
            break


BLR(2, 1, 1, [3, 2, 1])