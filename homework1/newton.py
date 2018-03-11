import numpy as np
import matrix as mat
import solve

def getHessionInv(features):
    # hession = 2(A^T)A
    hession = 2 * np.matmul(features.T, features)
    return mat.inv(hession)

def getGradient(features, weight, target):
    features = np.matrix(features)
    weight = np.matrix(weight)
    target = np.matrix(target)

    # gradient = 2(A^T)Ax - 2(A^T)b
    gradient = 2 * features.T * features * weight - 2 * features.T * target
    return np.array(gradient)

def newtonmethod(hession_inv, gradient, weight):
    return weight - np.matmul(hession_inv, gradient)


def optimize(x, y, base):
    weight = np.zeros((base ,1), dtype=float)
    target = np.array(y).reshape(-1, 1)
    features = solve.genfeatures(x, base)
    hession_inv = getHessionInv(features)
    gradient = getGradient(features, weight, target)
    weight = newtonmethod(hession_inv, gradient, weight)
    error = solve.geterror(weight, x, y)
    return weight, error