import numpy as np


def rosenbrock(Xs: np.matrix, a=1, b=100) -> np.matrix:
    return (a - Xs[0, :]) ** 2 + b * (Xs[1, :] - Xs[0, :] ** 2) ** 2


def rastrigin(Xs: np.matrix, A=10) -> np.matrix:
    N = Xs.shape[0]
    return A * N + np.sum(Xs ** 2 - A * np.cos(2 * np.pi * Xs), axis=0)
