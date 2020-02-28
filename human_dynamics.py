import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import urllib.request
import warnings

from matplotlib.backends import backend_gtk3
from pylatex import Document, Package
from pylatex.utils import NoEscape


# NOTE with the current method, SINDy (like other complex networks methods) can't detect edge weights


warnings.filterwarnings('ignore', module=backend_gtk3.__name__)


# Input and Output Settings
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')
UCI_ONLINE_URL = 'http://konect.uni-koblenz.de/downloads/tsv/opsahl-ucsocial.tar.bz2'
UCI_ONLINE_TAR_PATH = os.path.join(DATA_DIR, 'opsahl-ucsocial.tar.bz2')
UCI_ONLINE_DIR = os.path.join(DATA_DIR, 'opsahl-ucsocial')
UCI_ONLINE_TSV_PATH = os.path.join(UCI_ONLINE_DIR, 'out.opsahl-ucsocial')
TEMPORAL_BUCKET_SIZE = 3 * 60 * 60  # in seconds
ADJACENCY_MATRIX_PATH = os.path.join(DATA_DIR, 'adjacency_matrix.npy')
X_PATH = os.path.join(DATA_DIR, 'x.npy')


# Algorithm Settings
EPSILON = 10 ** -24
CROSS_VALIDATION_PERCENTAGE = 0.3  # range: [0, 1]
SINDY_ITERATIONS = 10
POWERS = [i * 0.5 for i in range(1, 5)]  # NOTE: there shouldn't be any zero powers
LAMBDA_RANGE = [-2000, 2]
LAMBDA_STEP = 1.1


# Calculated Settings
CANDIDATE_LAMBDAS = [
    LAMBDA_STEP ** i for i in range(
        -1 * int(math.log(abs(LAMBDA_RANGE[0])) / math.log(LAMBDA_STEP)),
        1 * int(math.log(abs(LAMBDA_RANGE[1])) / math.log(LAMBDA_STEP))
    )
]


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


def _get_data_matrices():
    first_timestamp = 0
    last_timestamp = 0
    min_id = sys.maxsize
    max_id = 0
    for from_id, to_id, count, timestamp in _data_generator():
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

    adjacency_matrix = np.zeros((number_of_users, number_of_users))
    x = np.zeros((number_of_buckets, number_of_users))
    for from_id, to_id, count, timestamp in _data_generator():
        adjacency_matrix[from_id - min_id, to_id - min_id] = 1
        bucket = int((timestamp - first_timestamp) / TEMPORAL_BUCKET_SIZE)
        x[bucket, from_id - min_id] += count

    cross_validation_index = int((1 - CROSS_VALIDATION_PERCENTAGE) * number_of_buckets)

    return adjacency_matrix, x[:cross_validation_index], x[cross_validation_index:]


def _get_x_dot(x):
    x_dot = (x[1:] - x[:len(x) - 1])
    return x_dot


def _get_theta(x, adjacency_matrix, node_index):
    number_of_nodes = x.shape[1]
    time_frames = x.shape[0] - 1
    theta = []
    latex_functions = []
    for j in range(time_frames):
        entry = [1]
        latex_functions.append(r'1')
        for first_power in POWERS:
            entry.append(x[j, node_index] ** first_power)
            if j == 0:
                latex_functions.append(r'x_{%d}^{%f}' % (node_index, first_power))
            for second_power in POWERS:
                entry.append(
                    sum([
                        adjacency_matrix[k, node_index] * (x[j, node_index] ** first_power) * (x[j, k] ** second_power)
                        for k in range(number_of_nodes) if k != node_index
                    ])
                )
                if j == 0:
                    latex_functions.append(
                        r'(' +
                        '+'.join([
                            r'%f * x_{%d}^{%f} * x_{%d}^{%f}' % (
                                adjacency_matrix[k, node_index], node_index, first_power, k, second_power
                            )
                            for k in range(number_of_nodes) if k != node_index
                        ]) +
                        r')'
                    )
        theta.append(entry)
    return np.array(theta), latex_functions


def _sindy(x_dot, theta, candidate_lambda):
    xi = np.linalg.lstsq(theta, x_dot, rcond=None)[0]
    for j in range(SINDY_ITERATIONS):
        small_indices = np.flatnonzero(np.absolute(xi) < candidate_lambda)
        big_indices = np.flatnonzero(np.absolute(xi) >= candidate_lambda)
        xi[small_indices] = 0
        xi[big_indices] = np.linalg.lstsq(theta[:, big_indices], x_dot, rcond=None)[0]
    return xi


def run():
    adjacency_matrix, x, x_cv = _get_data_matrices()
    x_dot = _get_x_dot(x)
    x_dot_cv = _get_x_dot(x_cv)
    number_of_nodes = x.shape[1]

    # SINDy for individual nodes
    latex_document = Document('basic')
    latex_document.packages.append(Package('breqn'))
    theta_list = []
    theta_cv_list = []
    first_node_latex_functions = None
    for node_index in range(number_of_nodes):
        print('### SINDy for node %d' % node_index)  # TODO remove
        theta, latex_functions = _get_theta(x, adjacency_matrix, node_index)
        theta_list.append(theta)
        if node_index == 0:
            first_node_latex_functions = latex_functions
        theta_cv, latex_functions = _get_theta(x_cv, adjacency_matrix, node_index)
        theta_cv_list.append(theta_cv)
        mse_list = []
        complexity_list = []
        least_cost = sys.maxsize
        best_xi = None
        selected_lambda = 0
        selected_complexity = 0
        selected_mse = 0
        for candidate_lambda in CANDIDATE_LAMBDAS:
            ith_derivative = x_dot[:, node_index]
            xi = _sindy(ith_derivative, theta, candidate_lambda)
            complexity = np.count_nonzero(xi) / np.prod(xi.shape)
            mse_cv = np.square(x_dot_cv[:, node_index] - (np.matmul(theta_cv, xi.T))).mean()
            mse_list.append(math.log10(EPSILON + mse_cv))
            complexity_list.append(complexity)

            if complexity:  # zero would mean no statements
                cost = mse_cv * complexity
                if cost < least_cost:
                    least_cost = cost
                    best_xi = xi
                    selected_lambda = candidate_lambda
                    selected_complexity = complexity
                    selected_mse = mse_cv

        plt.clf()
        plt.figure(figsize=(16, 9), dpi=96)
        plt.plot(complexity_list, mse_list)
        counter = {}
        for i in range(len(complexity_list)):
            complexity = complexity_list[i]
            mse_cv = mse_list[i]
            key = '%f_%f' % (complexity, mse_cv)
            if key in counter:
                counter[key][2] += 1
            else:
                counter[key] = [complexity, mse_cv, 1]
        for value in counter.values():
            plt.annotate(value[2], (value[0], value[1]))
        plt.title('lambda = %f, complexity = %f, log10(mse) = %f' % (
            selected_lambda,
            selected_complexity,
            math.log10(EPSILON + selected_mse)
        ))
        plt.xlabel('complexity (percentage of nonzero entries)')
        plt.ylabel('log10 of cross validation mean squared error')
        plt.savefig(os.path.join(OUTPUT_DIR, 'node_%d_lambda.png' % node_index))

        latex_document.append(NoEscape(r'\clearpage $'))
        line = r'\frac{dx_{%d}}{dt}=' % node_index
        line_content = []
        for j in range(best_xi.shape[0]):
            if best_xi[j]:
                line_content.append(r'%f' % best_xi[j] + latex_functions[j])

        line += ' + '.join(line_content)
        latex_document.append(NoEscape(line))
        latex_document.append(NoEscape(r'$'))
    latex_document.generate_pdf(os.path.join(OUTPUT_DIR, 'individual_equations.pdf'))

    # SINDy for the entire system
    entire_theta = np.concatenate(theta_list)
    entire_theta_cv = np.concatenate(theta_cv_list)
    entire_derivatives = np.concatenate([x_dot[:, node_index] for node_index in range(number_of_nodes)])
    entire_derivatives_cv = np.concatenate([x_dot_cv[:, node_index] for node_index in range(number_of_nodes)])
    mse_list = []
    complexity_list = []
    least_cost = sys.maxsize
    best_xi = None
    selected_lambda = 0
    selected_complexity = 0
    selected_mse = 0
    for candidate_lambda in CANDIDATE_LAMBDAS:
        entire_xi = _sindy(entire_derivatives, entire_theta, candidate_lambda)
        complexity = np.count_nonzero(entire_xi) / np.prod(entire_xi.shape)
        mse_cv = np.square(entire_derivatives_cv - (np.matmul(entire_theta_cv, entire_xi.T))).mean()
        mse_list.append(math.log10(EPSILON + mse_cv))
        complexity_list.append(complexity)

        if complexity:  # zero would mean no statements
            cost = mse_cv * complexity
            if cost < least_cost:
                least_cost = cost
                best_xi = entire_xi
                selected_lambda = candidate_lambda
                selected_complexity = complexity
                selected_mse = mse_cv

    plt.clf()
    plt.figure(figsize=(16, 9), dpi=96)
    plt.plot(complexity_list, mse_list)
    counter = {}
    for i in range(len(complexity_list)):
        complexity = complexity_list[i]
        mse_cv = mse_list[i]
        key = '%f_%f' % (complexity, mse_cv)
        if key in counter:
            counter[key][2] += 1
        else:
            counter[key] = [complexity, mse_cv, 1]
    for value in counter.values():
        plt.annotate(value[2], (value[0], value[1]))
    plt.title('lambda = %f, complexity = %f, log10(mse) = %f' % (
        selected_lambda,
        selected_complexity,
        math.log10(EPSILON + selected_mse)
    ))
    plt.xlabel('complexity (percentage of nonzero entries)')
    plt.ylabel('log10 of cross validation mean squared error')
    plt.savefig(os.path.join(OUTPUT_DIR, 'entire_system_lambda.png'))

    latex_document = Document('basic')
    latex_document.packages.append(Package('breqn'))
    latex_document.append(NoEscape(r'\clearpage $'))
    line = r'\frac{dx_0}{dt}='  # using first node as example
    line_content = []
    for j in range(best_xi.shape[0]):
        if best_xi[j]:
            line_content.append(r'%f' % best_xi[j] + first_node_latex_functions[j])

    line += ' + '.join(line_content)
    latex_document.append(NoEscape(line))
    latex_document.append(NoEscape(r'$'))
    latex_document.generate_pdf(os.path.join(OUTPUT_DIR, 'entire_system_equation.pdf'))


if __name__ == '__main__':
    run()
