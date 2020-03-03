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


# File and Directory Settings
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')
UCI_ONLINE_URL = 'http://konect.uni-koblenz.de/downloads/tsv/opsahl-ucsocial.tar.bz2'
UCI_ONLINE_TAR_PATH = os.path.join(DATA_DIR, 'opsahl-ucsocial.tar.bz2')
UCI_ONLINE_DIR = os.path.join(DATA_DIR, 'opsahl-ucsocial')
UCI_ONLINE_TSV_PATH = os.path.join(UCI_ONLINE_DIR, 'out.opsahl-ucsocial')


# Algorithm Settings
TEMPORAL_BUCKET_SIZE = 24 * 60 * 60  # in seconds
NODE_LIMIT = 150  # had most nonzero ratio experimentally
EPSILON = 10 ** -24
CROSS_VALIDATION_PERCENTAGE = 0.3  # range: [0, 1]
SINDY_ITERATIONS = 10
POWERS = {  # use constants instead of zero powers
    'f_constant': True,
    'f': np.arange(0.5, 2.5, 0.5).tolist(),
    'g_constant': True,  # this could be redundant if f_constant is true
    'g_solo_i': np.arange(0.5, 2.5, 0.5).tolist(),  # this could be redundant if f overlaps
    'g_solo_j': np.arange(0.5, 2.5, 0.5).tolist(),
    'g_combined_i': np.arange(0.5, 2.5, 0.5).tolist(),
    'g_combined_j': np.arange(0.5, 2.5, 0.5).tolist(),
}
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

    x = x + EPSILON

    cross_validation_index = int((1 - CROSS_VALIDATION_PERCENTAGE) * number_of_buckets)

    return adjacency_matrix, x[:cross_validation_index], x[cross_validation_index:]


def _get_x_dot(x):
    x_dot = (x[1:] - x[:len(x) - 1])
    return x_dot


def _get_theta(x, adjacency_matrix, node_index):
    number_of_nodes = x.shape[1]
    time_frames = x.shape[0] - 1
    x_i = x[:time_frames, node_index]
    adjacency_sum = np.sum(adjacency_matrix[:, node_index])
    column_list = []
    latex_functions = []

    if POWERS['f_constant']:
        column_list.append(np.ones(time_frames))
        latex_functions.append(r'1')

    for f_power in POWERS['f']:
        column_list.append(x_i ** f_power)
        latex_functions.append(r'x_{%d}^{%f}' % (node_index, f_power))

    if POWERS['g_constant']:
        column_list.append(np.full(time_frames, adjacency_sum))
        latex_functions.append(r'(\sum_j A_{j,%d})' % node_index)

    for g_solo_i_power in POWERS['g_solo_i']:
        column_list.append(adjacency_sum * x_i ** g_solo_i_power)
        latex_functions.append(r'(\sum_j A_{j,%d} * x_{%d}^{%f})' % (node_index, node_index, g_solo_i_power))

    for g_solo_j_power in POWERS['g_solo_j']:
        terms = []
        for j in range(number_of_nodes):
            if j != node_index and adjacency_matrix[j, node_index]:
                x_j = x[:time_frames, j]
                terms.append(adjacency_matrix[j, node_index] * x_j ** g_solo_j_power)
        if terms:
            column = np.sum(terms, axis=0)
            column_list.append(column)
            latex_functions.append(
                r'(\sum_j A_{j,%d} * x_j^{%f})' % (node_index, g_solo_j_power)
            )

    for g_combined_i_power in POWERS['g_combined_i']:
        for g_combined_j_power in POWERS['g_combined_j']:
            terms = []
            for j in range(number_of_nodes):
                if j != node_index and adjacency_matrix[j, node_index]:
                    x_j = x[:time_frames, j]
                    terms.append(adjacency_matrix[j, node_index] * x_i ** g_combined_i_power * x_j ** g_combined_j_power)
            if terms:
                column = np.sum(terms, axis=0)
                column_list.append(column)
                latex_functions.append(
                    r'(\sum_j A_{j,%d} * x_{%d}^{%f} * x_j^{%f})' % (
                        node_index, node_index, g_combined_i_power, g_combined_j_power
                    )
                )

    theta = np.column_stack(column_list)
    return theta, latex_functions


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
        theta_cv, latex_functions = _get_theta(x_cv, adjacency_matrix, node_index)
        theta_cv_list.append(theta_cv)
        if node_index == 0:
            first_node_latex_functions = latex_functions
        mse_list = []
        complexity_list = []
        least_cost = sys.maxsize
        best_xi = None
        selected_lambda = 0
        selected_complexity = 0
        selected_mse = 0
        ith_derivative = x_dot[:, node_index]
        for candidate_lambda in CANDIDATE_LAMBDAS:
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
        plt.close('all')

        latex_document.append(NoEscape(r'\clearpage $'))
        line = r'\frac{dx_{%d}}{dt}=' % node_index
        if best_xi is not None:
            line_content = []
            for j in range(best_xi.shape[0]):
                if best_xi[j]:
                    line_content.append(r'%f' % best_xi[j] + latex_functions[j])

            line += ' + '.join(line_content)
        else:
            line += '0'
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
    plt.close('all')

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
