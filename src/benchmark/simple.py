#!/usr/bin/env python3

import numpy as np


def rosenbrock(Xs: np.array, a=1, b=100) -> np.array:
    if (Xs.ndim > 1):
        x0 = Xs[0, :]
        x1 = Xs[1, :]
    else:
        x0 = Xs[0]
        x1 = Xs[1]

    return (a - x0) ** 2 + b * (x1 - x0 ** 2) ** 2


def rastrigin(Xs: np.array, A=10) -> np.array:
    N = Xs.shape[0]
    return A * N + np.sum(Xs ** 2 - A * np.cos(2 * np.pi * Xs), axis=0)


def shubert(Xs: np.array) -> np.array:
    x = Xs[0].reshape(-1)
    y = Xs[1].reshape(-1)

    i = np.arange(1, 6).reshape(-1, 1)

    return np.sum(
        i * np.cos((i + 1) * x + i), axis=0
    ) * np.sum(
        i * np.cos((i + 1) * y + 1), axis=0
    )
