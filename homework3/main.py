import data
import numpy as np

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
    point = data.polyBasisModel(basis, a, weight)
    prior_mean = 0
    prior_var_inv = b
estimator(20, 3)