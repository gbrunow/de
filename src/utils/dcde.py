#!/usr/bin/env python3

import numpy as np
from . import de, population
# from .population.selection import select


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
    popReductionEnabled: bool = True,
    returnPopulation: bool = False,
    subcomps: int = None,
    maxsubgens: int = None,
    controlDiversity: bool = False,
    alpha: float = 0.06, d=0.1, zeta=1,
    conquerMutationFactor: float = None,
    log: bool = False,
) -> np.array:

    if(len(boundaries) > 1 or dimensions is None):
        boundaries = np.array(boundaries)
        dimensions = boundaries.shape[0]
    else:
        boundaries = np.array(boundaries * dimensions)

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

    dividedPopulation = np.zeros((dimensions, populationSize * subcomps))

    for sc in range(subcomps):
        subcompStart = sc * populationSize
        subcompEnd = (sc + 1) * populationSize

        partialPopulation = optimizeGroup(
            populationSize=populationSize,
            boundaries=subcompBounds,
            mutationFactor=mutationFactor,
            crossingRate=crossingRate,
            fitness=fitness,
            maxgens=maxsubgens,
            subcomponent=sc,
            stopDiff=stopDiff,
        )

        dividedPopulation[:, subcompStart:subcompEnd] = partialPopulation

    dividedFitness = fitness(dividedPopulation)
    # --------------- end of divide --------------- #

    # ------------------ conquer ------------------ #
    dividedCandidates = dividedPopulation[  # get N best agents from divide population
        :,
        dividedFitness.argsort()[:populationSize]
    ]

    conquerPopulation = dividedCandidates

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
        returnPopulation=True
    )
