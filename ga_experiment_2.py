import math
import numpy as np
import random


# experiment 2: static number of statements, coefficients
CHROMOSOME_SIZE = 5
MUTATION_CHANCE = 0.01
MUTATION_RANGE = 1
POPULATION = 100
CHILDREN = 10
ITERATIONS = 100000
POWER_RANGE = (0, 4)
COEFFICIENT_RANGE = (1, 5)
EPSILON = 10 ** -8
# TODO experiment 3: dynamic number of statements, coefficients
# TODO remove chromosome size
# TODO add complexity to fitness
# TODO experiment 4: two variables
# TODO data-driven discovery of complex dynamical networks using intermittent genetic and SINDy


def _gauss_decay(value, origin, offset, scale, decay):
    """
    https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-function-score-query.html
    """
    sigma2 = -1 * scale ** 2 / (2 * math.log(decay))
    return math.exp(-1 * max(0, abs(value - origin) - offset) ** 2 / (2 * sigma2))


def _linear_decay(value, origin, offset, scale, decay):
    """
    https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-function-score-query.html
    """
    s = scale / (1 - decay)
    return max(0, (s - max(0, abs(value - origin) - offset)) / s)


class Individual:
    def __init__(self, chromosome, x, y):
        self.coefficients = None
        self.chromosome = chromosome
        self.fitness = self._calculate_fitness(x, y)

    def _calculate_mse(self, x, y):
        library = np.column_stack([x ** gene for gene in self.chromosome])
        coefficients = np.linalg.lstsq(library, y, rcond=None)[0]
        self.coefficients = coefficients
        y_hat = np.matmul(library, coefficients.T)
        return np.mean((y - y_hat) ** 2)

    def _calculate_distance_variance(self):  # incentivize equidistant powers
        sorted_chromosome = np.sort(self.chromosome)
        return np.var(sorted_chromosome[1:] - sorted_chromosome[:-1])

    def _calculate_least_difference(self):
        sorted_chromosome = np.sort(self.chromosome)
        return np.min(sorted_chromosome[1:] - sorted_chromosome[:-1])

    def _calculate_decay(self):
        total_decay = 1

        chromosome_origin = (POWER_RANGE[1] + POWER_RANGE[0]) / 2
        chromosome_offset = (POWER_RANGE[1] - POWER_RANGE[0]) / 2
        chromosome_scale = 1
        chromosome_decay = 0.5
        for gene in self.chromosome:
            total_decay *= _linear_decay(
                gene,
                chromosome_origin,
                chromosome_offset,
                chromosome_scale,
                chromosome_decay
            )

        coefficient_origin = (COEFFICIENT_RANGE[1] + COEFFICIENT_RANGE[0]) / 2
        coefficient_offset = (COEFFICIENT_RANGE[1] - COEFFICIENT_RANGE[0]) / 2
        coefficient_scale = 1
        coefficient_decay = 0.5
        for coefficient in self.coefficients:
            total_decay *= _linear_decay(
                coefficient,
                coefficient_origin,
                coefficient_offset,
                coefficient_scale,
                coefficient_decay
            )

        return EPSILON + (1 - EPSILON) * total_decay

    def _calculate_fitness(self, x, y):
        mse = self._calculate_mse(x, y)
        decay = self._calculate_decay()
        return decay / mse


class Population:
    def __init__(self, size, x, y):
        self.size = size
        self.x = x
        self.y = y
        self.individuals = self._initialize_individuals()

    def _initialize_individuals(self):
        individuals = []
        for i in range(self.size):
            individuals.append(Individual(
                [POWER_RANGE[0] + random.random() * (POWER_RANGE[1] - POWER_RANGE[0]) for _ in range(CHROMOSOME_SIZE)],
                self.x,
                self.y
            ))
        return individuals

    def _crossover(self, individual1, individual2):
        offspring_chromosome = []
        for i in range(CHROMOSOME_SIZE):
            offspring_chromosome.append(random.choice(
                (individual1.chromosome[i], individual2.chromosome[i])
            ))
        return Individual(offspring_chromosome, self.x, self.y)

    def _mutation(self, individual):
        mutated_chromosome = []
        for i in range(CHROMOSOME_SIZE):
            if random.random() < MUTATION_CHANCE:
                mutated_chromosome.append(individual.chromosome[i] + (random.random() - 0.5) * MUTATION_RANGE)
            else:
                mutated_chromosome.append(individual.chromosome[i])
        return Individual(mutated_chromosome, self.x, self.y)

    def run_single_iteration(self):
        sorted_individuals = sorted(self.individuals, key=lambda individual: -1 * individual.fitness)

        children = []
        for i in range(CHILDREN):
            children.append(self._mutation(self._crossover(sorted_individuals[i], sorted_individuals[i + 1])))

        new_individuals = sorted(self.individuals + children, key=lambda individual: -1 * individual.fitness)
        self.individuals = new_individuals[:self.size]

        return self.individuals[0]  # fittest


def _get_x():
    return 1 + np.random.rand(10) * 10


def _get_y(x):
    result = 0
    for i in range(CHROMOSOME_SIZE):
        result += (i + 1) * x ** i
    return result


def run():
    x = _get_x()
    y = _get_y(x)
    population = Population(POPULATION, x, y)
    fittest_individual = None
    for i in range(ITERATIONS):
        fittest_individual = population.run_single_iteration()
        if i % 1000 == 0:
            print(1 / fittest_individual.fitness)
    print(fittest_individual.chromosome)
    print(fittest_individual.coefficients)


if __name__ == '__main__':
    run()
