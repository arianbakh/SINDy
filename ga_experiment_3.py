import numpy as np
import random


# experiment 3: static number of statements, coefficients, discrete genes


DATA_POINTS = 100
CHROMOSOME_SIZE = 5
GENE_SIZE = 12  # bits
MUTATION_CHANCE = 0.1
POPULATION = 100
CHILDREN = 10
ITERATIONS = 1000000
POWER_RANGE = (0, 5)


STEP = (POWER_RANGE[1] - POWER_RANGE[0]) / 2 ** GENE_SIZE


class Individual:
    def __init__(self, chromosome, x, y):
        self.coefficients = None
        self.chromosome = chromosome
        self.fitness = self._calculate_fitness(x, y)

    def _calculate_mse(self, x, y):
        powers = []
        for i in range(CHROMOSOME_SIZE):
            binary = 0
            for j in range(GENE_SIZE):
                binary += self.chromosome[i * GENE_SIZE + j] * 2 ** (GENE_SIZE - j - 1)
            power = POWER_RANGE[0] + binary * STEP
            powers.append(power)
        self.powers = powers
        library = np.column_stack([x ** power for power in self.powers])
        coefficients = np.linalg.lstsq(library, y, rcond=None)[0]
        self.coefficients = coefficients
        y_hat = np.matmul(library, coefficients.T)
        return np.mean((y - y_hat) ** 2)

    def _calculate_least_difference(self):
        sorted_powers = np.sort(self.powers)
        return np.min(sorted_powers[1:] - sorted_powers[:-1])

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
                [random.randint(0, 1) for _ in range(CHROMOSOME_SIZE * GENE_SIZE)],
                self.x,
                self.y
            ))
        return individuals

    def _crossover(self, individual1, individual2):
        crossover_point = random.randint(0, CHROMOSOME_SIZE * GENE_SIZE - 1)
        offspring_chromosome = individual1.chromosome[:crossover_point] + individual2.chromosome[crossover_point:]
        return Individual(offspring_chromosome, self.x, self.y)

    def _mutation(self, individual):
        mutated_chromosome = []
        for i in range(CHROMOSOME_SIZE * GENE_SIZE):
            if random.random() < MUTATION_CHANCE:
                mutated_chromosome.append(0 if individual.chromosome[i] else 1)
            else:
                mutated_chromosome.append(individual.chromosome[i])
        return Individual(mutated_chromosome, self.x, self.y)

    @staticmethod
    def _select_random_individual(sorted_individuals, total_fitness):
        random_value = random.random()
        selected_index = 0
        selected_individual = sorted_individuals[0]
        sum_fitness = selected_individual.fitness
        for i in range(1, len(sorted_individuals)):
            if sum_fitness / total_fitness > random_value:
                break
            selected_index = i
            selected_individual = sorted_individuals[i]
            sum_fitness += selected_individual.fitness
        return selected_index, selected_individual

    def run_single_iteration(self):
        # the following two values are pre-calculated to increase performance
        sorted_individuals = sorted(self.individuals, key=lambda individual: -1 * individual.fitness)
        total_fitness = sum([individual.fitness for individual in self.individuals])

        children = []
        while len(children) < CHILDREN:
            individual1_index, individual1 = self._select_random_individual(sorted_individuals, total_fitness)
            individual2_index, individual2 = self._select_random_individual(sorted_individuals, total_fitness)
            if individual1_index != individual2_index:
                children.append(self._mutation(self._crossover(individual1, individual2)))

        new_individuals = sorted(self.individuals + children, key=lambda individual: -1 * individual.fitness)
        self.individuals = new_individuals[:self.size]

        return self.individuals[0]  # fittest


def _get_x():
    return 1 + np.random.rand(DATA_POINTS) * 10


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
    print(fittest_individual.powers)
    print(fittest_individual.coefficients)


if __name__ == '__main__':
    run()
