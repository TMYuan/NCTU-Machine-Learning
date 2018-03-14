import numpy as np
import matrix as mat
from sklearn.preprocessing import PolynomialFeatures

def genfeatures(x, degree):
    """
    Generate features from original input data.
    e.g.
    degree = 3
    [1 x x^2]
    """
    features = []
    for point in x:
        features.append([point**base for base in range(degree)])
    return np.array(features)

def genmatrix(features, rate):
    """
    input: A, rate
    output: (A^T)(A) - (rate)(I)
    This function will return (A^T)(A) - rate * I
    """
    matrix = mat.mul(mat.transpose(features), features)
    matrix -= rate * np.eye(matrix.shape[0])
    return matrix

def genb(features, y):
    """
    input: A, b
    output: (A^T)(b)
    """
    y = np.array(y).reshape(-1, 1)
    b = mat.mul(mat.transpose(features), y)
    return b

def geterror(weight, x, y):
    """
    input: weight, input data, target
    output: square error
    This function will calculate square error between prediction and target.
    """
    base = weight.shape[0]
    weight_new = weight.flatten()
    error = 0.0

    for (x_, y_) in zip(x, y):
        predict = 0.0
        for b in range(base):
            predict += weight_new[b] * (x_**b)
            # print(predict)
        error += (predict - y_) ** 2
    return error

def LSE(x, y, base, rate):
    features = genfeatures(x, base)
    print(features)
    matrix = genmatrix(features, rate)
    b = genb(features, y)
    L, U = mat.LUdecomposition(matrix)
    weight = mat.LUsolver(L, U, b)
    error = geterror(weight, x, y)
    return weight, error

if __name__ == '__main__':
    # A = np.array([
    #     [3., -1., 2.],
    #     [6., -1., 5.],
    #     [-9., 7., 3.]
    # ])
    # b = np.array([[10., 22., -7.]])
    # L, U = LUdecomposition(A)
    # print(L)
    # print(U)

    # x = LUsolver(L, U, b)
    weight = np.array([[2.], [3.], [2]])
    x = [3, 7]
    y = [5, 10]
    error = geterror(weight, x, y)
    print(error)
