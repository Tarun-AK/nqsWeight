import numpy as np


def lrSampler(L):
    np.random.seed(10)
    samples = np.random.choice([-1, 1], size = (100000, L))
    samples[:, L-2] = samples[:, 1]
    return samples



