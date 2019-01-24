from . import de
import numpy as np
from joblib import Parallel, delayed
import multiprocessing


def optimize(
    populationSize: int,
    boundaries: np.array,
    mutationFactor: float,
    crossingRate: float,
    fitness: callable,
    generations: int,
    dimensions=None,
    controlDiversity: bool = False,
    alpha: float = 0.06, d=0.1, zeta=1,
    parallel: bool = False
) -> np.array:

    if(len(boundaries) > 1 or dimensions is None):
        boundaries = np.array(boundaries)
        dimensions = boundaries.shape[0]
    else:
        boundaries = np.array(boundaries * dimensions)

    # ---- divide ---- #
    groups = 10
    groupGenerations = generations / groups

    groupBounds = np.zeros((dimensions, groups + 1))
    for d in range(boundaries.shape[0]):
        groupBounds[d, :] = np.linspace(
            boundaries[d, 0], boundaries[d, 1], groups + 1)

    if(parallel):
        cores = multiprocessing.cpu_count()

        dividedPopulation = Parallel(n_jobs=cores * 3, prefer="threads")(
            delayed(optimizeGroup)(
                populationSize=populationSize,
                boundaries=groupBounds,
                mutationFactor=mutationFactor,
                crossingRate=crossingRate,
                fitness=fitness,
                generations=groupGenerations,
                group=g,
            ) for g in range(groups)
        )

        dividedPopulation = np.array(dividedPopulation).transpose(1, 0, 2).reshape(
            dimensions, groups * populationSize
        )
    else:
        dividedPopulation = np.zeros((dimensions, populationSize * groups))

        for g in range(groups):
            groupStart = g * populationSize
            groupEnd = (g + 1) * populationSize

            partialPopulation = optimizeGroup(
                populationSize=populationSize,
                boundaries=groupBounds,
                mutationFactor=mutationFactor,
                crossingRate=crossingRate,
                fitness=fitness,
                generations=groupGenerations,
                group=g
            )

            dividedPopulation[:, groupStart:groupEnd] = partialPopulation

    dividedFitness = fitness(dividedPopulation)
    # ---- conquer ---- #

    # get N best agents from divide population
    conquerPopulation = dividedPopulation[
        :,
        dividedFitness.argsort()[:populationSize]
    ]

    return de.optimize(
        populationSize=populationSize,
        boundaries=boundaries,
        mutationFactor=mutationFactor,
        crossingRate=crossingRate,
        fitness=fitness,
        generations=groupGenerations,
        population=conquerPopulation,
    )


def optimizeGroup(
    populationSize: int,
    boundaries: np.array,
    mutationFactor: float,
    crossingRate: float,
    fitness: callable,
    generations: int,
    group: int
) -> np.array:
    bounds = boundaries[:, group: (group + 2)]

    return de.optimize(
        populationSize=populationSize,
        boundaries=bounds,
        mutationFactor=mutationFactor,
        crossingRate=crossingRate,
        fitness=fitness,
        generations=generations,
        returnPopulation=True
    )
