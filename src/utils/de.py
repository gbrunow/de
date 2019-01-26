import numpy as np
from .diversity import control


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


def optimize(
    populationSize: int,
    boundaries: np.array,
    mutationFactor: float,
    crossingRate: float,
    fitness: callable,
    maxgens: int = None,
    stopDiff: float = 1e-5,
    dimensions: int = None,
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
            trials = control(
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

        generation = generation + 1

        if (generation == maxgens or diff <= stopDiff):
            if(log):
                print(f'finished at generation {generation}, diff {diff}')
            break

    if(returnPopulation):
        return pop
    else:
        evaluation = evaluate(agents=pop, fitness=fitness)
        best = pop[:, np.argmin(evaluation)]

        # return [best, allTimeBest]
        return best


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


def evaluate(agents: np.array, fitness) -> np.array:
    return fitness(agents)


def normalize(data: np.array) -> np.array:
    maximum = np.max(data)
    minimum = np.min(data)

    den = maximum - minimum
    if(den == 0):
        den = 1

    return (data - minimum)/den
