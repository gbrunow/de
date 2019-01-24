from . import de
import numpy as np


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

    dividePopulation = np.zeros((dimensions, populationSize * groups))
    for g in range(groups):
        bounds = groupBounds[:, g: (g + 2)]
        groupStart = g * populationSize
        groupEnd = (g + 1) * populationSize

        partialPopulation = de.optimize(
            populationSize=populationSize,
            boundaries=bounds,
            mutationFactor=mutationFactor,
            crossingRate=crossingRate,
            fitness=fitness,
            generations=groupGenerations,
            returnPopulation=True
        )

        dividePopulation[:, groupStart:groupEnd] = partialPopulation

    divideFitness = fitness(dividePopulation)
    # ---- conquer ---- #

    # get N best agents from divide population
    conquerPopulation = dividePopulation[
        :,
        divideFitness.argsort()[:populationSize]
    ]

    return de.optimize(
        populationSize=populationSize,
        boundaries=bounds,
        mutationFactor=mutationFactor,
        crossingRate=crossingRate,
        fitness=fitness,
        generations=groupGenerations,
        population=conquerPopulation,
    )
