import random

def gaussianGenerator(mean, variance):
    """
    This function will return sampling point from a given gaussian distribution.
    """
    total = 0
    for _ in range(12):
        total += random.uniform(0, 1)
    z = total - 6
    return mean + z * (variance**0.5)

def polyBasisModel(basis, variance, weight):
    """
    This function will return a target point with a given linear polynomial model.
    """
    x = random.uniform(-10, 10)
    y = gaussianGenerator(0, variance)
    for i in range(basis):
        y += weight[i] * (x ** basis)
    return y

