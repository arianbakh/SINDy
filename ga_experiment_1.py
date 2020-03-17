import numpy as np
import random


# experiment 1: static number of statements, no coefficients
CHROMOSOME_SIZE = 5
MUTATION_CHANCE = 0.01
MUTATION_RANGE = 1
POPULATION = 100
CHILDREN = 10
ITERATIONS = 100000


class Individual:
    def __init__(self, chromosome, x, y):
        self.chromosome = chromosome
        self.fitness = self._calculate_fitness(x, y)

    def _calculate_mse(self, x, y):
        y_hat = 0
        for i in range(CHROMOSOME_SIZE):
            y_hat += x ** self.chromosome[i]
        return np.mean((y - y_hat) ** 2)

    def _calculate_distance_variance(self):  # incentivize equidistant powers
        sorted_chromosome = np.sort(self.chromosome)
        return np.var(sorted_chromosome[1:] - sorted_chromosome[:-1])

    def _calculate_least_difference(self):
        sorted_chromosome = np.sort(self.chromosome)
        return np.min(sorted_chromosome[1:] - sorted_chromosome[:-1])

    def _calculate_fitness(self, x, y):
        mse = self._calculate_mse(x, y)
        least_difference = self._calculate_least_difference()
        return least_difference / mse


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
                [random.random() for _ in range(CHROMOSOME_SIZE)],
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
    return np.random.rand(10)


def _get_y(x):
    result = 0
    for i in range(CHROMOSOME_SIZE):
        result += x ** i
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


if __name__ == '__main__':
    run()
