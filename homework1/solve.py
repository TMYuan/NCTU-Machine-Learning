import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def genfeatures(x, degree):
    features = []
    for point in x:
        features.append([point**base for base in range(degree)])
    return np.array(features)

def genmatrix(features, rate):
    """
    input: A, rate
    output: (A^T)(A) - (rate)(I)
    This function will return (A^T)(A)
    """
    matrix = np.matmul(features.T, features)
    matrix -= rate * np.eye(matrix.shape[0])
    return matrix

def genb(features, y):
    """
    input: A, b
    output: (A^T)(b)
    """
    y = np.array(y).reshape(-1, 1)
    b = np.matmul(features.T, y)
    return b

def LUdecomposition(a):
    """
    input: A
    output: L, U
    """
    assert a.shape[0] == a.shape[1]
    n = a.shape[0]
    L = np.eye(n)
    # print(L)
    U = a.copy()
    for i in range(n-1):
        pivot = U[i][i]
        L[i+1:, i] = U[i+1:, i] / pivot
        for j in range(i+1, n):
            U[j, :] = U[j, :] - L[j][i] * U[i, :]
    return (L, U)

def LUsolver(L, U, b):
    """
    input: L, U, b
    output: x
    """
    n = L.shape[0]
    y = np.zeros((n, 1))
    x = np.zeros((n, 1))

    # Solve y vector
    for i in range(n):
        y[i][0] = b[i][0]
        if i != 0:
            for j in range(i):
                y[i][0] += (-L[i][j]) * y[j][0]
    # print(y)

    # Solve x vector
    for i in reversed(range(n)):
        x[i][0] = y[i][0]
        if i != n-1:
            for j in reversed(range(i+1 ,n)):
                x[i][0] += (-U[i][j]) * x[j][0]
        x[i][0] /= U[i][i]
    # print(x)
    return x

def geterror(weight, x, y):
    base = weight.shape[0]
    weight_new = weight.flatten()
    error = 0.0

    for (x_, y_) in zip(x, y):
        predict = 0.0
        for b in range(base):
            predict += weight_new[b] * (x_**b)
            print(predict)
        error += (predict - y_) ** 2
    return error

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
