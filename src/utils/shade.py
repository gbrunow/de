# #!/usr/bin/env python3

import numpy as np
from .population import diversity, reduction
from .population.common import populate, mutate, cross, evaluate, select, best, shouldStop, processBoundaries, validate, result, archive


def optimize(
    populationSize: int,
    boundaries: np.array,
    mutationFactor: float,
    crossingRate: float,
    fitness: callable,
    maxgens: int = None,
    stopDiff: float = 1e-5,
    dimensions: int = None,
    minPopulationSize: int = 4,
    reducePopulation: bool = True,
    archiveSize: int = None,
    population: np.array = None,
    returnPopulation: bool = False,
    log: bool = False,
) -> np.array:
    '''
    `returns` global minimum of the objective function

    Parameters
    ------------
    `populationSize`:       
        Size of the population that will perform the search.
        Must be greater or equals to `4`.
    `boundaries`:           
        Defines the search boundaries.
        Array of `(min, max)`, required for each dimension 
            OR
        Array containing 1 `(min, max)` and `dimensions`, 
        this will carry over same `boundary` to every dimension.
    `mutationFactor`:       
        Mutation factor.
    `crossingRate`:         
        Crossing rate.
    `fitness`:              
        Objective function.
    `maxgens`:          
        Maximum number of generations.
    `stopDiff`:
        Diff stop criteria.
        Stop if difference between worst and best trial vectors
        is smaller than `stopDiff`.
        default = `1e-5`
    `dimensions` (optional):            
        Problem dimension. 
        Will be calculate using `boundaries` if `len(boundaries) > 1`
    `minPopulationSize` (optional):
        Minimum population size used by the population reduction algorithm.
        Must be greater or equals to `4`.
        default = `4`
    `reducePopulation` (optional):
        Enables population size reduction.
        default = `True`
    `archiveSize` (optional):
        Defines the size of the external archive.
        default = `populationSize`
    `returnPopulation` (optional):
        Set to `True` to get population instead of best agent
        default = `False`
    `population` (optional):
        Initialized population.
        Will be randomly generated if not provided (recomended).
    '''

    [valid, errors] = validate(
        populationSize,
        population,
        minPopulationSize
    )

    if(not valid):
        raise ValueError('Input parameters are not valid', errors)

    if (archiveSize is None):
        archiveSize = populationSize

    [boundaries, dimensions] = processBoundaries(boundaries, dimensions)

    if(population is None):
        pop = populate(
            size=populationSize,
            boundaries=boundaries,
            dimensions=dimensions
        )
    else:
        pop = population

    externalArchive = None

    generation = 0
    while True:
        mutants = mutate(population=pop, factor=mutationFactor)
        trials = cross(population=pop, mutants=mutants, rate=crossingRate)

        [selected, evaluation, rejected] = select(
            population=pop,
            trials=trials,
            fitness=fitness
        )

        pop = selected

        externalArchive = archive(
            externalArchive=externalArchive,
            rejected=rejected,
            archiveSize=archiveSize
        )

        if (reducePopulation):
            pop = reduction.linear(
                population=pop,
                size=populationSize,
                minSize=minPopulationSize,
                generation=generation,
                maxGenerations=maxgens,
                evaluation=evaluation
            )

        generation = generation + 1

        if (shouldStop(evaluation, stopDiff, generation, maxgens)):
            if(log):
                print(f'finished at generation {generation}')
            break

    return result(
        returnPopulation=returnPopulation,
        population=pop,
        evaluation=evaluation
    )
