#!/usr/bin/env python3

import numpy as np
from .common import best


def linear(
    population: np.array,
    evaluation: np.array,
    size: int,
    minSize: int,
    generation: int,
    maxGenerations: int
):
    currentSize = population.shape[1]
    newSize = round(
        size +
        (
            (minSize - size) / maxGenerations
        ) * generation
    )

    if (newSize < currentSize):
        return best(
            population=population,
            evaluation=evaluation,
            n=newSize
        )
    else:
        return population
