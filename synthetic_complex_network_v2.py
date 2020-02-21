import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import warnings

from matplotlib.backends import backend_gtk3
from pylatex import Document, Package
from pylatex.utils import NoEscape


# TODO one regression to rule them all -> concat
# TODO adaptive lambda
# TODO test random weights
# TODO anything other than complexity and mse for cost?
# TODO plot the same plot in lorentz
# TODO LASSO
# NOTE with the current method, SINDy (like other complex networks methods) can't detect edge weights


warnings.filterwarnings('ignore', module=backend_gtk3.__name__)


# Directory Settings
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


# Algorithm Settings
NUMBER_OF_NODES = 5
DATA_FRAMES = 1000
DELTA_T = 0.01
SINDY_ITERATIONS = 10
POWERS = [i * 0.5 for i in range(1, 5)] + [i * -0.5 for i in range(1, 5)]  # NOTE: there shouldn't be any zero powers
LAMBDA_RANGE = [-2000, 2]
LAMBDA_STEP = 1.1


# Calculated Settings
CANDIDATE_LAMBDAS = [
    LAMBDA_STEP ** i for i in range(
        -1 * int(math.log(abs(LAMBDA_RANGE[0])) / math.log(LAMBDA_STEP)),
        1 * int(math.log(abs(LAMBDA_RANGE[1])) / math.log(LAMBDA_STEP))
    )
]


def _get_adjacency_matrix():
    a = np.zeros((NUMBER_OF_NODES, NUMBER_OF_NODES))
    for i in range(NUMBER_OF_NODES):
        for j in range(NUMBER_OF_NODES):
            if i != j:
                a[i, j] = 1.33
                # a[i, j] = random.random()
    return a


def _get_x(a, time_frames):
    x = np.zeros((time_frames + 1, NUMBER_OF_NODES))
    x[0] = np.array(
        [random.random() * i * 1000 for i in range(1, NUMBER_OF_NODES + 1)]
    )  # NOTE: values must be large enough and different
    for i in range(1, time_frames + 1):
        for j in range(NUMBER_OF_NODES):
            f_result = -1 * (x[i - 1, j] ** 1.5)
            g_result = 0
            for k in range(NUMBER_OF_NODES):
                if k != j:
                    g_result += a[k, j] * (x[i - 1, j] ** 0.5) * (x[i - 1, k] ** 0.5)
            derivative = f_result + g_result
            x[i, j] = x[i - 1, j] + DELTA_T * derivative
    return x


def _get_x_dot(x):
    x_dot = (x[1:] - x[:len(x) - 1]) / DELTA_T
    return x_dot


def _get_theta(x, adjacency_matrix, node_index):
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
                        for k in range(NUMBER_OF_NODES) if k != node_index
                    ])
                )
                if j == 0:
                    latex_functions.append(
                        r'(' +
                        '+'.join([
                            r'%f * x_{%d}^{%f} * x_{%d}^{%f}' % (
                                adjacency_matrix[k, node_index], node_index, first_power, k, second_power
                            )
                            for k in range(NUMBER_OF_NODES) if k != node_index
                        ]) +
                        r')'
                    )
        theta.append(entry)
    return np.array(theta), latex_functions


def _sindy(x_dot, theta, candidate_lambda, node_index):
    ith_derivative = x_dot[:, node_index]
    xi = np.linalg.lstsq(theta, ith_derivative, rcond=None)[0]
    for j in range(SINDY_ITERATIONS):
        small_indices = np.flatnonzero(np.absolute(xi) < candidate_lambda)
        big_indices = np.flatnonzero(np.absolute(xi) >= candidate_lambda)
        xi[small_indices] = 0
        xi[big_indices] = np.linalg.lstsq(theta[:, big_indices], ith_derivative, rcond=None)[0]
    return xi


def run():
    adjacency_matrix = _get_adjacency_matrix()
    x = _get_x(adjacency_matrix, DATA_FRAMES)
    x_cv = _get_x(adjacency_matrix, int(DATA_FRAMES / 2))
    x_dot = _get_x_dot(x)
    x_dot_cv = _get_x_dot(x_cv)

    latex_document = Document('basic')
    latex_document.packages.append(Package('breqn'))
    for node_index in range(NUMBER_OF_NODES):
        theta, latex_functions = _get_theta(x, adjacency_matrix, node_index)
        theta_cv, latex_functions = _get_theta(x_cv, adjacency_matrix, node_index)
        mse_list = []
        complexity_list = []
        least_cost = sys.maxsize
        best_xi = None
        selected_lambda = 0
        selected_complexity = 0
        selected_mse = 0
        for candidate_lambda in CANDIDATE_LAMBDAS:
            xi = _sindy(x_dot, theta, candidate_lambda, node_index)
            complexity = np.count_nonzero(xi) / np.prod(xi.shape)
            mse_cv = np.square(x_dot_cv[:, node_index] - (np.matmul(theta_cv, xi.T))).mean()
            mse_list.append(math.log10(mse_cv))
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
            math.log10(selected_mse)
        ))
        plt.xlabel('complexity (percentage of nonzero entries)')
        plt.ylabel('log10 of cross validation mean squared error')
        plt.savefig(os.path.join(OUTPUT_DIR, 'node_%d_mse_complexity.png' % node_index))

        latex_document.append(NoEscape(r'\clearpage $'))
        line = r'\frac{dx_{%d}}{dt}=' % node_index
        line_content = []
        for j in range(best_xi.shape[0]):
            if best_xi[j]:
                line_content.append(r'%f' % best_xi[j] + latex_functions[j])

        line += ' + '.join(line_content)
        latex_document.append(NoEscape(line))
        latex_document.append(NoEscape(r'$'))
    latex_document.generate_pdf(os.path.join(OUTPUT_DIR, 'equations.pdf'))


if __name__ == '__main__':
    run()
