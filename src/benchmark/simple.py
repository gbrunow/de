import numpy as np


def rosenbrock(Xs: np.matrix, a=1, b=100) -> np.matrix:
    if (Xs.ndim > 1):
        x0 = Xs[0, :]
        x1 = Xs[1, :]
    else:
        x0 = Xs[0]
        x1 = Xs[1]

    return (a - x0) ** 2 + b * (x1 - x0 ** 2) ** 2


def rastrigin(Xs: np.matrix, A=10) -> np.matrix:
    N = Xs.shape[0]
    return A * N + np.sum(Xs ** 2 - A * np.cos(2 * np.pi * Xs), axis=0)
