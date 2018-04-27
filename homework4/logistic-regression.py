import numpy as np
import pandas as pd
import data

def getdata(mean, var, n):
    dataset = []
    for _ in range(n):
        x = data.gaussianGenerator(mean[0], var[0])
        y = data.gaussianGenerator(mean[1], var[1])
        dataset.append((x, y))
    return np.array(dataset)

dataset1 = getdata((10, 3), (1, 5), 100)
dataset2 = getdata((20, 1), (5, 1), 100)
print(dataset1.shape)
print(dataset2.shape)