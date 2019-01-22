import numpy as np


def populate(size: int, min: float, max: float, dimensions: int) -> np.matrix:
    if (size < 4):
        raise ValueError('Population size has to be greater than 4')
    if (min >= max):
        raise ValueError('Parameter "max" needs to be greater than "min"')

    return (max - min) * np.random.rand(dimensions, size) + min


def mutate(population: np.matrix, factor: float) -> np.matrix:
    if (factor <= 0):
        raise ValueError('Mutation factor has to be greater than zero.')

    mutations = getMutations(population=population, factor=factor)
    return population + mutations


def cross(population: np.matrix, mutants: np.matrix, rate: float) -> np.matrix:
    cross = getWillCross(population=population, rate=rate)
    trials = np.zeros(population.shape)
    trials[cross] = mutants[cross]
    trials[~cross] = population[~cross]

    return trials


def select(population: np.matrix, trials: np.matrix, fitness) -> np.matrix:
    popScore = evaluate(agents=population, fitness=fitness)
    trialScore = evaluate(agents=trials, fitness=fitness)

    # copying to avoiding side effects, could use reference to improve performance
    # some testing to compare performance impact may be necessary
    generation = np.copy(trials)
    betterParents = popScore < trialScore
    generation[:, betterParents] = population[:, betterParents]

    return generation


def optimize(
    populationSize: int,
    dimentions: int,
    min: float,
    max: float,
    mutationFactor: float,
    crossingRate: float,
    fitness,
    generations: int = 1000,
) -> np.array:
    pop = populate(
        size=populationSize,
        min=min, max=max,
        dimensions=dimentions
    )
    generation = 0
    while True:
        mutants = mutate(population=pop, factor=mutationFactor)
        trials = cross(population=pop, mutants=mutants, rate=crossingRate)
        pop = select(population=pop, trials=trials, fitness=fitness)

        generation = generation + 1
        if (generation == generations):
            break

    evaluation = evaluate(agents=pop, fitness=fitness)
    bestIndex = np.argmin(evaluation)
    best = pop[:, bestIndex]

    return best


def getMutations(population: np.matrix, factor: float) -> np.matrix:
    NP = population.shape[1]

    a = np.random.randint(NP, size=NP)
    b = np.random.randint(NP, size=NP)

    repeated = a == b
    while(np.any(repeated)):
        np.place(b, repeated, np.random.randint(NP))
        repeated = a == b

    mutation = factor * (population[:, a] - population[:, b])
    return mutation


def getWillCross(population: np.matrix, rate: float) -> np.matrix:
    [dimensions, size] = population.shape
    willCross = np.random.rand(dimensions, size) <= rate

    for d in range(dimensions):
        forceCross = np.random.randint(0, dimensions, size) == d
        willCross[d, :] = willCross[d, :] | forceCross

    return willCross


def evaluate(agents: np.matrix, fitness) -> np.array:
    return fitness(agents)
