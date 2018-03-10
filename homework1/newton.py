import numpy as np
import solve

def getHessionInv(features):
    # hession = 2(A^T)A
    hession = 2 * np.matmul(features.T, features)
    return np.linalg.inv(hession)

def getGradient(features, weight, target):
    features = np.matrix(features)
    weight = np.matrix(weight)
    target = np.matrix(target)

    # gradient = 2(A^T)Ax - 2(A^T)b
    gradient = 2 * features.T * features * weight - 2 * features.T * target
    return np.array(gradient)

def newtonmethod(hession_inv, gradient, weight):
    return weight - np.matmul(hession_inv, gradient)

