import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def genfeatures(x, degree):
    features = []
    for point in x:
        features.append([point**base for base in range(degree)])
    return np.array(features)

def genmatrix(features):
    matrix = np.matmul(features.T, features)
    return matrix

def genb(features, y):
    y = np.array(y).reshape(-1, 1)
    b = np.matmul(features.T, y)
    return b

def LUdecomposition(a):
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
    n = L.shape[0]
    y = np.zeros((1, n))
    x = np.zeros((1, n))

    # Solve y vector
    for i in range(n):
        y[0][i] = b[0][i]
        if i != 0:
            for j in range(i):
                y[0][i] += (-L[i][j]) * y[0][j]
    print(y)

    # Solve x vector
    for i in reversed(range(n)):
        x[0][i] = y[0][i]
        if i != n-1:
            for j in reversed(range(i+1 ,n)):
                x[0][i] += (-U[i][j]) * x[0][j]
        x[0][i] /= U[i][i]
    print(x)
    return x

if __name__ == '__main__':
    A = np.array([
        [3., -1., 2.],
        [6., -1., 5.],
        [-9., 7., 3.]
    ])
    b = np.array([[10., 22., -7.]])
    L, U = LUdecomposition(A)
    print(L)
    print(U)

    x = LUsolver(L, U, b)
