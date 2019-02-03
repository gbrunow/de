#!/usr/bin/env python3

import numpy as np
from .population import diversity, reduction
from .population.common import populate, mutate, cross, evaluate, select, best


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
    popReductionEnabled: bool = True,
    population: np.array = None,
    returnPopulation: bool = False,
    controlDiversity: bool = False,
    alpha: float = 0.06, d=0.1, zeta=1,
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
    `enablePopulationReduction` (optional):
        Enables population size reduction.
        default = `True`
    `returnPopulation` (optional):
        Set to `True` to get population instead of best agent
        default = `False`
    `population` (optional):
        Initialized population.
        Will be randomly generated if not provided (recomended).
    `controlDiversity` (optional):      
        Set to `True` to enable diversity control.
        default = `False`
    `alpha` (optional):                 
        Diversity control parameter.
    `d` (optional):                     
        Diversity control parameter.
    `zeta` (optional):                  
        Diversity control parameter.
    '''

    if (populationSize < 4 and population is None):
        raise ValueError(
            'Parameter `populationSize` must be greater or equals to 4.'
        )

    if (minPopulationSize < 4):
        raise ValueError(
            'Parameter `minPopulationSize` must be greater or equals to 4.'
        )

    if(population is None):
        pop = populate(
            size=populationSize,
            boundaries=boundaries,
            dimensions=dimensions
        )
    else:
        pop = population

    # popArchive = None
    # fitnessArchive = None
    # allTimeBest = None
    # allTimeBestFitness = None

    generation = 0
    while True:
        mutants = mutate(population=pop, factor=mutationFactor)
        trials = cross(population=pop, mutants=mutants, rate=crossingRate)

        if (controlDiversity):
            trials = diversity.control(
                population=pop,
                trials=trials,
                g=generation,
                n=maxgens,
                alpha=alpha, d=d, zeta=zeta
            )

        [pop, evaluation] = select(
            population=pop,
            trials=trials,
            fitness=fitness
        )

        # popArchive = concat(new=pop, archive=popArchive)
        # fitnessArchive = concat(new=evaluation, archive=fitnessArchive)

        currBest = np.min(evaluation)
        currWorst = np.max(evaluation)
        diff = currWorst - currBest

        # if (allTimeBestFitness is None or currBest < allTimeBestFitness):
        #     allTimeBestFitness = currBest
        #     allTimeBest = pop[:, np.argmin(evaluation)]

        if (popReductionEnabled):
            reduction.linear(
                population=pop,
                size=populationSize,
                minSize=minPopulationSize,
                generation=generation,
                maxGenerations=maxgens,
                evaluation=evaluation
            )

        generation = generation + 1

        if (generation == maxgens or diff <= stopDiff):
            if(log):
                print(f'finished at generation {generation}, diff {diff}')
            break

    if(returnPopulation):
        return pop
    else:
        evaluation = evaluate(agents=pop, fitness=fitness)
        # best = pop[:, np.argmin(evaluation)]

        # return [best, allTimeBest]
        return best(population=pop, evaluation=evaluation, n=1)
