import numpy as np
import os
import random
import sys
import tarfile
import urllib.request


# experiment 5: static number of statements, coefficients, discrete genes, human dynamics (USocial data)


# File and Directory Settings
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')
UCI_ONLINE_URL = 'http://konect.uni-koblenz.de/downloads/tsv/opsahl-ucsocial.tar.bz2'
UCI_ONLINE_TAR_PATH = os.path.join(DATA_DIR, 'opsahl-ucsocial.tar.bz2')
UCI_ONLINE_DIR = os.path.join(DATA_DIR, 'opsahl-ucsocial')
UCI_ONLINE_TSV_PATH = os.path.join(UCI_ONLINE_DIR, 'out.opsahl-ucsocial')


# Input Data Settings
TEMPORAL_BUCKET_SIZE = 24 * 60 * 60  # in seconds  # originally 3 hours
NODE_LIMIT = 100  # 150 had most nonzero ratio experimentally
CROSS_VALIDATION_PERCENTAGE = 0.3  # range: [0, 1]


# Genetic Settings
CHROMOSOME_SIZE = 4
GENE_SIZE = 12  # bits
MUTATION_CHANCE = 0.1
POPULATION = 100
CHILDREN = 10
ITERATIONS = 100
POWER_RANGE = (-1, 4)


# Calculated Settings
STEP = (POWER_RANGE[1] - POWER_RANGE[0]) / 2 ** GENE_SIZE


class Individual:
    def __init__(self, chromosome, x, y, adjacency_matrix):
        self.coefficients = None
        self.chromosome = chromosome
        self.fitness = self._calculate_fitness(x, y, adjacency_matrix)

    def _get_theta(self, x, adjacency_matrix):
        number_of_nodes = x.shape[1]
        time_frames = x.shape[0] - 1

        theta_list = []
        for node_index in range(number_of_nodes):
            x_i = x[:time_frames, node_index]
            adjacency_sum = np.sum(adjacency_matrix[:, node_index])
            column_list = [
                np.ones(time_frames),
                x_i ** self.powers[0],
                adjacency_sum * x_i ** self.powers[1],
            ]
            terms = []
            for j in range(number_of_nodes):
                if j != node_index and adjacency_matrix[j, node_index]:
                    x_j = x[:time_frames, j]
                    terms.append(
                        adjacency_matrix[j, node_index] * x_i ** self.powers[2] * x_j ** self.powers[3])
            if terms:
                column = np.sum(terms, axis=0)
                column_list.append(column)
            theta = np.column_stack(column_list)
            theta_list.append(theta)
        return np.concatenate(theta_list)

    def _calculate_mse(self, x, y, adjacency_matrix):
        number_of_nodes = x.shape[1]

        powers = []
        for i in range(CHROMOSOME_SIZE):
            binary = 0
            for j in range(GENE_SIZE):
                binary += self.chromosome[i * GENE_SIZE + j] * 2 ** (GENE_SIZE - j - 1)
            power = POWER_RANGE[0] + binary * STEP
            powers.append(power)
        self.powers = powers
        theta = self._get_theta(x, adjacency_matrix)
        stacked_y = np.concatenate([y[:, node_index] for node_index in range(number_of_nodes)])
        coefficients = np.linalg.lstsq(theta, stacked_y, rcond=None)[0]
        self.coefficients = coefficients
        y_hat = np.matmul(theta, coefficients.T)
        return np.mean((stacked_y - y_hat) ** 2)

    def _calculate_least_difference(self):
        sorted_powers = np.sort(self.powers)
        return np.min(sorted_powers[1:] - sorted_powers[:-1])

    def _calculate_fitness(self, x, y, adjacency_matrix):
        mse = self._calculate_mse(x, y, adjacency_matrix)
        least_difference = self._calculate_least_difference()
        return least_difference / mse


class Population:
    def __init__(self, size, x, y, adjacency_matrix):
        self.size = size
        self.x = x
        self.y = y
        self.adjacency_matrix = adjacency_matrix
        self.individuals = self._initialize_individuals()

    def _initialize_individuals(self):
        individuals = []
        for i in range(self.size):
            individuals.append(Individual(
                [random.randint(0, 1) for _ in range(CHROMOSOME_SIZE * GENE_SIZE)],
                self.x,
                self.y,
                self.adjacency_matrix
            ))
        return individuals

    def _crossover(self, individual1, individual2):
        crossover_point = random.randint(0, CHROMOSOME_SIZE * GENE_SIZE - 1)
        offspring_chromosome = individual1.chromosome[:crossover_point] + individual2.chromosome[crossover_point:]
        return Individual(offspring_chromosome, self.x, self.y, self.adjacency_matrix)

    def _mutation(self, individual):
        mutated_chromosome = []
        for i in range(CHROMOSOME_SIZE * GENE_SIZE):
            if random.random() < MUTATION_CHANCE:
                mutated_chromosome.append(0 if individual.chromosome[i] else 1)
            else:
                mutated_chromosome.append(individual.chromosome[i])
        return Individual(mutated_chromosome, self.x, self.y, self.adjacency_matrix)

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


def _ensure_data():
    if not os.path.exists(UCI_ONLINE_DIR):
        urllib.request.urlretrieve(UCI_ONLINE_URL, UCI_ONLINE_TAR_PATH)
        tar = tarfile.open(UCI_ONLINE_TAR_PATH, "r:bz2")
        tar.extractall(DATA_DIR)
        tar.close()


def _data_generator():
    _ensure_data()
    with open(UCI_ONLINE_TSV_PATH, 'r') as tsv_file:
        for i, line in enumerate(tsv_file.readlines()):
            if not line.startswith('%'):
                split_line = line.strip().split()
                from_id = int(split_line[0])
                to_id = int(split_line[1])
                count = int(split_line[2])
                timestamp = int(split_line[3])
                yield from_id, to_id, count, timestamp


def _normalize_x(x):
    return x + 1  # TODO


def _get_data_matrices():
    first_timestamp = 0
    last_timestamp = 0
    min_id = sys.maxsize
    max_id = 0
    edge_count = {}
    for from_id, to_id, count, timestamp in _data_generator():
        key = '%d_%d' % (min(from_id, to_id), max(from_id, to_id))
        if key not in edge_count:
            edge_count[key] = 0
        edge_count[key] += 1

        if not first_timestamp:
            first_timestamp = timestamp
        last_timestamp = timestamp
        if from_id > max_id:
            max_id = from_id
        if to_id > max_id:
            max_id = to_id
        if from_id < min_id:
            min_id = from_id
        if to_id < min_id:
            min_id = to_id
    number_of_users = max_id - min_id + 1
    number_of_buckets = int((last_timestamp - first_timestamp) / TEMPORAL_BUCKET_SIZE) + 1

    selected_nodes = set()
    for edge, count in sorted(edge_count.items(), key=lambda item: -item[1]):
        if len(selected_nodes) < min(NODE_LIMIT, number_of_users):
            involved_nodes = {int(item) for item in edge.split('_')}
            selected_nodes = selected_nodes.union(involved_nodes)
        else:
            break
    new_index = {}
    for i, selected_node in enumerate(selected_nodes):
        new_index[selected_node] = i
    number_of_nodes = len(selected_nodes)

    adjacency_matrix = np.zeros((number_of_nodes, number_of_nodes))
    x = np.zeros((number_of_buckets, number_of_nodes))
    for from_id, to_id, count, timestamp in _data_generator():
        if from_id in selected_nodes and to_id in selected_nodes:
            adjacency_matrix[new_index[from_id], new_index[to_id]] = 1
            bucket = int((timestamp - first_timestamp) / TEMPORAL_BUCKET_SIZE)
            x[bucket, new_index[from_id]] += count

    cross_validation_index = int((1 - CROSS_VALIDATION_PERCENTAGE) * number_of_buckets)

    x = _normalize_x(x)

    return adjacency_matrix, x[:cross_validation_index], x[cross_validation_index:]


def _get_y(x):
    x_dot = (x[1:] - x[:len(x) - 1])
    return x_dot


def run():
    adjacency_matrix, x, x_cv = _get_data_matrices()
    y = _get_y(x)
    population = Population(POPULATION, x, y, adjacency_matrix)
    fittest_individual = None
    for i in range(ITERATIONS):
        fittest_individual = population.run_single_iteration()
        print(1 / fittest_individual.fitness)
    print('%f + %f * xi^%f + %f * sum Aij xi^%f + %f * sum Aij * xi^%f * xj^%f' % (
        fittest_individual.coefficients[0],
        fittest_individual.coefficients[1],
        fittest_individual.powers[0],
        fittest_individual.coefficients[2],
        fittest_individual.powers[1],
        fittest_individual.coefficients[3],
        fittest_individual.powers[2],
        fittest_individual.powers[3]
    ))


if __name__ == '__main__':
    run()
