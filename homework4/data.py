import random
import numpy as np

def gaussianGenerator(mean, variance):
    """
    This function will return sampling point from a given gaussian distribution.
    """
    total = 0
    for _ in range(12):
        total += random.uniform(0, 1)
    z = total - 6
    return mean + z * (variance**0.5)
