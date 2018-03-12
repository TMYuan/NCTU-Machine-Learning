import numpy as np
import matrix as mat
import solve

def getHessionInv(features):
    # hession = 2(A^T)A
    features_T = mat.transpose(features)
    hession = 2 * mat.mul(features_T, features)
    return mat.inv(hession)

def getGradient(features, weight, target):
    # gradient = 2(A^T)Ax - 2(A^T)b
    # gradient = 2 * features.T * features * weight - 2 * features.T * target
    features_T = mat.transpose(features)
    gradient = 2 * mat.mul(mat.mul(features_T, features), weight)
    gradient -= 2 * mat.mul(features_T, target)
    return gradient

def newtonmethod(hession_inv, gradient, weight):
    # x1 = x0 - (H^-1)(gradient)
    return weight - mat.mul(hession_inv, gradient)


def optimize(x, y, base):
    weight = np.zeros((base ,1), dtype=float)
    target = np.array(y).reshape(-1, 1)
    features = solve.genfeatures(x, base)
    hession_inv = getHessionInv(features)
    gradient = getGradient(features, weight, target)
    weight = newtonmethod(hession_inv, gradient, weight)
    error = solve.geterror(weight, x, y)
    return weight, error