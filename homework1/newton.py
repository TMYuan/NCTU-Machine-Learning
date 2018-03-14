import numpy as np
import matrix as mat
import solve

def getHessionInv(features):
    """
    This function calculates hession matrix of LSE( 2(A^T)A )
    """
    # hession = 2(A^T)A
    features_T = mat.transpose(features)
    hession = 2 * mat.mul(features_T, features)
    return mat.inv(hession)

def getGradient(features, weight, target):
    """
    This function calculates gradient of LSE( 2(A^T)Ax - 2(A^T)b )
    """
    # gradient = 2(A^T)Ax - 2(A^T)b
    features_T = mat.transpose(features)
    gradient = 2 * mat.mul(mat.mul(features_T, features), weight)
    gradient -= 2 * mat.mul(features_T, target)
    return gradient

def newtonmethod(hession_inv, gradient, weight):
    """
    Using newton's method to optimize the answer.
    xn = xn-1 
    """
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