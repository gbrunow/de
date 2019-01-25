from . import de
import numpy as np


def optimize(
    populationSize: int,
    boundaries: np.array,
    mutationFactor: float,
    crossingRate: float,
    fitness: callable,
    generations: int,
    groups: int = None,
    groupGenerations: int = None,
    dimensions=None,
    controlDiversity: bool = False,
    alpha: float = 0.06, d=0.1, zeta=1,
    increaseMutation: bool = False,
    conquerMutationFactor: float = None,
) -> np.array:

    if(len(boundaries) > 1 or dimensions is None):
        boundaries = np.array(boundaries)
        dimensions = boundaries.shape[0]
    else:
        boundaries = np.array(boundaries * dimensions)

    if (groups is None):
        meanSearchAmplitude = np.sum(np.absolute(np.mean(boundaries, axis=0)))
        groups = int(round(max(meanSearchAmplitude / 10, 10)))

    # ---- divide ---- #
    if(groupGenerations is None):
        groupGenerations = round(generations / groups)

    groupBounds = np.zeros((dimensions, groups + 1))
    for d in range(boundaries.shape[0]):
        groupBounds[d, :] = np.linspace(
            boundaries[d, 0], boundaries[d, 1], groups + 1)

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
    dividedCandidates = dividedPopulation[
        :,
        dividedFitness.argsort()[:populationSize]
    ]

    conquerPopulation = dividedCandidates

    if (conquerMutationFactor is not None):
        mutationFactor = conquerMutationFactor
    elif(increaseMutation):
        mutationFactor = mutationFactor * 1.25

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
