import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def genfeatures(x, degree):
    x = np.array(x).reshape(-1, 1)
    poly = PolynomialFeatures(degree - 1)
    features = poly.fit_transform(x)
    return features

def genmatrix(features):
    matrix = np.matmul(features.T, features)
    return matrix

def genb(features, y):
    y = np.array(y).reshape(-1, 1)
    b = np.matmul(features.T, y)
    return b