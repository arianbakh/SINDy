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


# NOTE with the current method, SINDy (like other complex networks methods) can't detect edge weights


warnings.filterwarnings('ignore', module=backend_gtk3.__name__)


# Directory Settings
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


# Algorithm Settings
NUMBER_OF_NODES = 10
DATA_FRAMES = 1000
CV_DATA_FRAMES = 100
M = 10
DELTA_T = 0.01
SINDY_ITERATIONS = 10
POWERS = {  # use constants instead of zero powers
    'f_constant': True,
    'f': [0.5, 1, 1.5, 2],
    'g_constant': True,  # this could be redundant if f_constant is true
    'g_solo_i': [],  # this could be redundant if f overlaps
    'g_solo_j': [0.5, 1, 1.5, 2],
    'g_combined_i': [0.5, 1, 1.5, 2],
    'g_combined_j': [0.5, 1, 1.5, 2],
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


def _get_adjacency_matrix():
    a = np.zeros((NUMBER_OF_NODES, NUMBER_OF_NODES))
    for i in range(NUMBER_OF_NODES):
        for j in range(NUMBER_OF_NODES):
            if i != j:
                a[i, j] = random.random()
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
        for j in range(NUMBER_OF_NODES):
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
            for j in range(NUMBER_OF_NODES):
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
    adjacency_matrix = _get_adjacency_matrix()
    x = _get_x(adjacency_matrix, DATA_FRAMES)
    x_dot = _get_x_dot(x)
    x_cv_list = []
    x_dot_cv_list = []
    for observation in range(M):
        x_cv = _get_x(adjacency_matrix, CV_DATA_FRAMES)
        x_cv_list.append(x_cv)
        x_dot_cv = _get_x_dot(x_cv)
        x_dot_cv_list.append(x_dot_cv)

    # SINDy for individual nodes
    latex_document = Document('basic')
    latex_document.packages.append(Package('breqn'))
    for node_index in range(NUMBER_OF_NODES):
        theta, latex_functions = _get_theta(x, adjacency_matrix, node_index)
        aicc_list = []
        least_aicc = sys.maxsize
        best_xi = None
        ith_derivative = x_dot[:, node_index]
        for candidate_lambda in CANDIDATE_LAMBDAS:
            xi = _sindy(ith_derivative, theta, candidate_lambda)
            k = np.count_nonzero(xi)
            error = 0
            for observation in range(M):
                x_cv = x_cv_list[observation]
                x_dot_cv = x_dot_cv_list[observation]
                theta_cv, _ = _get_theta(x_cv, adjacency_matrix, node_index)
                error += np.sum(np.abs(x_dot_cv[:, node_index] - (np.matmul(theta_cv, xi.T))))
            aicc = M * math.log(error / M) + 2 * k
            if M - k - 2:
                aicc += 2 * (k + 1) * (k + 2) / (M - k - 2)
            else:  # TODO what to do with division by zero
                aicc += 2 * (k + 1) * (k + 2)
            aicc_list.append(aicc)
            if aicc < least_aicc:
                least_aicc = aicc
                best_xi = xi

        plt.figure(figsize=(16, 9), dpi=96)
        plt.plot([math.log10(candidate_lambda) for candidate_lambda in CANDIDATE_LAMBDAS], aicc_list)
        plt.xlabel('log10(lambda)')
        plt.ylabel('AIC')
        plt.savefig(os.path.join(OUTPUT_DIR, 'node_%d_lambda.png' % node_index))
        plt.close('all')

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


if __name__ == '__main__':
    run()
