#!/usr/bin/env python3

import numpy as np
from . import de
from .population.common import processBoundaries, concat, best


def optimize(
    populationSize: int,
    boundaries: np.array,
    mutationFactor: float,
    crossingRate: float,
    fitness: callable,
    maxgens: int = None,
    stopDiff: float = 1e-5,
    dimensions=None,
    minPopulationSize: int = 4,
    reducePopulation: bool = True,
    returnPopulation: bool = False,
    subcomps: int = None,
    maxsubgens: int = None,
    conquerMutationFactor: float = None,
    log: bool = False,
) -> np.array:

    [boundaries, dimensions] = processBoundaries(boundaries, dimensions)

    if (subcomps is None):
        meanSearchAmplitude = np.sum(np.absolute(np.mean(boundaries, axis=0)))
        subcomps = int(round(max(meanSearchAmplitude / 10, 10)))

    # ------------------ divide ------------------ #
    if(maxsubgens is None):
        maxsubgens = round(maxgens / subcomps)

    subcompBounds = np.zeros((dimensions, subcomps + 1))
    for d in range(boundaries.shape[0]):
        subcompBounds[d, :] = np.linspace(
            boundaries[d, 0], boundaries[d, 1], subcomps + 1)

    dividedPopulation = None

    for sc in range(subcomps):
        partialPopulation = optimizeGroup(
            populationSize=populationSize,
            boundaries=subcompBounds,
            mutationFactor=mutationFactor,
            crossingRate=crossingRate,
            fitness=fitness,
            maxgens=maxsubgens,
            subcomponent=sc,
            stopDiff=stopDiff,
            minPopulationSize=minPopulationSize,
            reducePopulation=reducePopulation,
        )

        dividedPopulation = concat(dividedPopulation, partialPopulation)

    dividedFitness = fitness(dividedPopulation)
    # --------------- end of divide --------------- #

    # ------------------ conquer ------------------ #
    conquerPopulation = best(
        population=dividedPopulation,
        evaluation=dividedFitness,
        n=populationSize
    )

    if (conquerMutationFactor is not None):
        mutationFactor = conquerMutationFactor

    return de.optimize(
        populationSize=populationSize,
        boundaries=boundaries,
        mutationFactor=mutationFactor,
        crossingRate=crossingRate,
        fitness=fitness,
        maxgens=maxgens,
        population=conquerPopulation,
        stopDiff=stopDiff,
        returnPopulation=returnPopulation,
        log=log
    )
    # -------------- end of conquer --------------- #


def optimizeGroup(
    populationSize: int,
    boundaries: np.array,
    mutationFactor: float,
    crossingRate: float,
    fitness: callable,
    maxgens: int,
    subcomponent: int,
    stopDiff: float,
    minPopulationSize: int,
    reducePopulation: bool,


) -> np.array:
    bounds = boundaries[:, subcomponent: (subcomponent + 2)]

    return de.optimize(
        populationSize=populationSize,
        boundaries=bounds,
        mutationFactor=mutationFactor,
        crossingRate=crossingRate,
        fitness=fitness,
        maxgens=maxgens,
        stopDiff=stopDiff,
        minPopulationSize=minPopulationSize,
        reducePopulation=reducePopulation,
        returnPopulation=True
    )
