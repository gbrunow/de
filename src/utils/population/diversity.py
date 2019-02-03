#!/usr/bin/env python3

import numpy as np


def control(population: np.array, trials: np.array, g: int, n: int, alpha: float = 0.06, d=0.1, zeta=1) -> np.array:
    threshhold = getThreshhold(g, n, alpha, d, zeta)

    movement = np.absolute(
        population - trials
    )
    controled = np.copy(trials)
    restore = movement < threshhold
    controled[restore] = population[restore]

    return controled


def getThreshhold(g: int, n: int, alpha: float = 0.06, d=0.1, zeta=1) -> float:
    """
    `returns` minimum vector movement threshold 
    Paremeters
    ----------
    `g`: current generation/iteration
    `n`: total number of iterations
    `alpha`: fraction of the main space diagonal
    `d`: initial value of threshold
    `zeta`: controls the decay rate of the threshold
    """
    if (alpha < 0 or alpha > 1):
        raise ValueError('Invalid alpha')

    return alpha * d * ((n - g)/n) ** zeta
