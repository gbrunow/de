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


def shubert(x: np.array, y: np.array) -> np.array:
    i = np.arange(1, 6).reshape(-1, 1)

    return np.sum(
        i * np.cos((i + 1) * x + i), axis=0
    ) * np.sum(
        i * np.cos((i + 1) * y + 1), axis=0
    )
