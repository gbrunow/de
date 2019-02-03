#!/usr/bin/env python3

import numpy as np


def populate(size: int, boundaries: np.array, dimensions: int = None) -> np.array:
    '''
    `returns` matrix containing randomly initialized population

    Parameters
    ------------
    `populationSize`:
        Size of the population that will perform the search.
    `boundaries`:           
        Defines the search boundaries.
        Array of `(min, max)`, required for each dimension 
            OR
        Array containing 1 `(min, max)` and `dimensions`, 
        this will carry over same `boundary` to every dimension.
    `dimensions` (optional):           
        Problem dimension. 
        Will be calculate using `boundaries` if `len(boundaries) > 1`
    '''

    if(len(boundaries) > 1 or dimensions is None):
        boundaries = np.array(boundaries)
        dimensions = boundaries.shape[0]
    else:
        boundaries = np.array(boundaries * dimensions)

    min = boundaries[:, 0].reshape(-1, 1)
    max = boundaries[:, 1].reshape(-1, 1)

    if (size < 4):
        raise ValueError('Population size has to be greater than 4')
    if (np.any(min >= max)):
        raise ValueError('Parameter "max" needs to be greater than "min"')

    return np.multiply((max - min), np.random.rand(dimensions, size)) + min


def mutate(population: np.array, factor: float) -> np.array:
    if (factor <= 0):
        raise ValueError('Mutation factor has to be greater than zero.')

    mutations = getMutations(population=population, factor=factor)
    return population + mutations


def cross(population: np.array, mutants: np.array, rate: float) -> np.array:
    cross = getWillCross(population=population, rate=rate)
    trials = np.copy(population)
    trials[cross] = mutants[cross]

    return trials


def evaluate(agents: np.array, fitness) -> np.array:
    return fitness(agents)


def select(population: np.array, trials: np.array, fitness) -> np.array:
    popScore = evaluate(agents=population, fitness=fitness)
    trialScore = evaluate(agents=trials, fitness=fitness)

    # copying to avoiding side effects, could use reference to improve performance
    # some testing to compare performance impact may be necessary
    generation = np.copy(population)
    improvements = trialScore < popScore
    generation[:, improvements] = trials[:, improvements]

    evaluation = np.copy(popScore)
    evaluation[improvements] = trialScore[improvements]

    return [generation, evaluation]


def concat(new: np.array, archive: np.array) -> np.array:
    if (archive is None):
        return new
    else:
        x = 0
        if (archive.ndim > 1):
            x = 1
        else:
            x = 0
        return np.concatenate([archive, new], axis=x)


def getMutations(population: np.array, factor: float) -> np.array:
    NP = population.shape[1]

    a = np.random.randint(NP, size=NP)
    b = np.random.randint(NP, size=NP)

    repeated = a == b
    while(np.any(repeated)):
        np.place(b, repeated, np.random.randint(NP))
        repeated = a == b

    mutation = factor * (population[:, a] - population[:, b])
    return mutation


def getWillCross(population: np.array, rate: float) -> np.array:
    [dimensions, size] = population.shape
    willCross = np.random.rand(dimensions, size) <= rate

    for d in range(dimensions):
        forceCross = np.random.randint(0, dimensions, size) == d
        willCross[d, :] = willCross[d, :] | forceCross

    return willCross


def best(population: np.array,  evaluation: np.array, n: int = 1) -> np.array:
    best = population[:, evaluation.argsort()[:n]]

    return best


def normalize(data: np.array) -> np.array:
    maximum = np.max(data)
    minimum = np.min(data)

    den = maximum - minimum
    if(den == 0):
        den = 1

    return (data - minimum)/den
