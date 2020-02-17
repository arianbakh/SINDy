import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys

from pylatex import Document, Package
from pylatex.utils import NoEscape


# TODO save final graph structure to file
# TODO one regression to rule them all
# TODO optimize speed


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
NUMBER_OF_NODES = 4
DATA_FRAMES = 1000
DELTA_T = 0.01
SINDY_ITERATIONS = 10
POWERS = np.arange(0.5, 2.5, 0.5).tolist()


def _get_adjacency_matrix():
    a = np.zeros((NUMBER_OF_NODES, NUMBER_OF_NODES))
    for i in range(NUMBER_OF_NODES):
        for j in range(NUMBER_OF_NODES):
            if i != j:
                a[i, j] = 1
                # a[i, j] = random.random()
    return a


def _get_x(a, time_frames):
    x = np.zeros((time_frames + 1, NUMBER_OF_NODES))
    x[0] = np.array([random.random() * 1000 for i in range(NUMBER_OF_NODES)])
    print(x[0])  # TODO remove
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


def _get_theta(x):
    theta = []
    time_frames = x.shape[0] - 1
    latex_functions = []
    for i in range(time_frames):
        entry = [1]
        latex_functions.append(r'1')
        for j in range(NUMBER_OF_NODES):
            for power in POWERS:
                entry.append(x[i, j] ** power)
                if i == 0:
                    latex_functions.append(r'x_{%d}^{%f}' % (j, power))
        for j in range(NUMBER_OF_NODES):
            for k in range(j + 1, NUMBER_OF_NODES):
                for first_power in POWERS:
                    for second_power in POWERS:
                        entry.append((x[i, j] ** first_power) * (x[i, k] ** second_power))
                        latex_functions.append(r'x_{%d}^{%f} x_{%d}^{%f}' % (j, first_power, k, second_power))
        theta.append(entry)
    return np.array(theta), latex_functions


def _sindy(x_dot, theta, candidate_lambda):
    xi = np.zeros((NUMBER_OF_NODES, theta.shape[1]))
    for i in range(NUMBER_OF_NODES):
        ith_derivative = x_dot[:, i]
        ith_xi = np.linalg.lstsq(theta, ith_derivative, rcond=None)[0]
        for j in range(SINDY_ITERATIONS):
            small_indices = np.flatnonzero(np.absolute(ith_xi) < candidate_lambda)
            big_indices = np.flatnonzero(np.absolute(ith_xi) >= candidate_lambda)
            ith_xi[small_indices] = 0
            ith_xi[big_indices] = np.linalg.lstsq(theta[:, big_indices], ith_derivative, rcond=None)[0]
        xi[i] = ith_xi
    return xi


def run():
    a = _get_adjacency_matrix()
    x = _get_x(a, DATA_FRAMES)
    x_cv = _get_x(a, int(DATA_FRAMES / 2))
    x_dot = _get_x_dot(x)
    x_dot_cv = _get_x_dot(x_cv)
    theta, latex_functions = _get_theta(x)
    theta_cv, latex_functions = _get_theta(x_cv)
    mse_list = []
    complexity_list = []
    least_cost = sys.maxsize
    best_xi = None
    for i in range(-12, 4, 1):
        candidate_lambda = 2 ** i
        xi = _sindy(x_dot, theta, candidate_lambda)
        complexity = np.count_nonzero(xi) / np.prod(xi.shape)
        mse_cv = np.square(x_dot_cv - (np.matmul(theta_cv, xi.T))).mean()
        mse_list.append(math.log10(mse_cv))
        complexity_list.append(complexity)

        if complexity:  # zero means no statements
            cost = mse_cv * complexity
            if cost < least_cost:
                least_cost = cost
                best_xi = xi

    plt.plot(complexity_list, mse_list)
    plt.xlabel('complexity (percentage of nonzero entries)')
    plt.ylabel('log10 of cross validation mean squared error')
    plt.savefig(os.path.join(OUTPUT_DIR, 'mse_complexity.png'))

    latex_document = Document('basic')
    latex_document.packages.append(Package('breqn'))
    for i in range(best_xi.shape[0]):
        latex_document.append(NoEscape(r'\begin{dmath}'))
        line = r'\frac{dx_{%d}}{dt}=' % i
        line_content = []
        for j in range(best_xi.shape[1]):
            if best_xi[i, j]:
                line_content.append(r'%f' % best_xi[i, j] + latex_functions[j])

        line += '+'.join(line_content)
        latex_document.append(NoEscape(line))
        latex_document.append(NoEscape(r'\end{dmath}'))
    latex_document.generate_pdf(os.path.join(OUTPUT_DIR, 'equations.pdf'))


if __name__ == '__main__':
    run()
