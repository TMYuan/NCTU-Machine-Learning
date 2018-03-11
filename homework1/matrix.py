import numpy as np

def mul(a, b):
    assert a.shape[1] == b.shape[0]
    n = a.shape[0]
    m = b.shape[1]
    r = a.shape[1]
    c = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            for k in range(r):
                c[i][j] += a[i][k] * b[k][j]
    return c

def transpose(a):
    assert len(a.shape) == 2
    n = a.shape[1]
    m = a.shape[0]
    b = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            b[i][j] = a[j][i]
    return b

def inv(a):
    assert a.shape[0] == a.shape[1]
    n = a.shape[0]
    b = np.zeros((n, n))
    L, U = LUdecomposition(a)
    for i in range(n):
        c = np.zeros((n, 1))
        c[i][0] = 1
        b[:, i] = LUsolver(L, U, c).flatten()
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

if __name__ == '__main__':
    # a = np.array([[1., 0., 0.], [2., 1., 0.], [-1., 2., 1.]])
    # b = np.array([[2., 2., 3.], [0., 3., 1.], [-0., 0., 6.]])
    # c = mul(a, b)
    # print(c)
    # d = transpose(a)
    # print(d)
    a = np.array([[1., 2.], [3., 4.]])
    print(inv(a))