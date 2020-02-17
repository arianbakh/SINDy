import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random


# TODO save final differential equation (latex) and graph structure to file
# TODO one regression to rule them all


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
NUMBER_OF_NODES = 10
DELTA_T = 0.01
SINDY_ITERATIONS = 10
POWERS = np.arange(0.5, 2.5, 0.5).tolist()


def _get_adjacency_matrix():
    a = np.zeros((NUMBER_OF_NODES, NUMBER_OF_NODES))
    for i in range(NUMBER_OF_NODES):
        for j in range(NUMBER_OF_NODES):
            if i != j:
                # a[i, j] = random.random()
                a[i, j] = 1
    return a


def _get_x(a, time_frames):
    x = np.zeros((time_frames + 1, NUMBER_OF_NODES))
    x[0] = np.array([random.random() for i in range(NUMBER_OF_NODES)])
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
    for i in range(time_frames):
        entry = [1]
        for j in range(NUMBER_OF_NODES):
            for power in POWERS:
                entry.append(x[i, j] ** power)
        for j in range(NUMBER_OF_NODES):
            for k in range(j + 1, NUMBER_OF_NODES):
                for first_power in POWERS:
                    for second_power in POWERS:
                        entry.append((x[i, j] ** first_power) * (x[i, k] ** second_power))
        theta.append(entry)
    return np.array(theta)


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
    x = _get_x(a, 100)
    x_cv = _get_x(a, 50)
    x_dot = _get_x_dot(x)
    x_dot_cv = _get_x_dot(x_cv)
    theta = _get_theta(x)
    theta_cv = _get_theta(x_cv)
    mse_list = []
    complexity_list = []
    for i in range(-12, 4, 1):
        candidate_lambda = 2 ** i
        xi = _sindy(x_dot, theta, candidate_lambda)
        complexity = np.count_nonzero(xi) / np.prod(xi.shape)
        mse_cv = np.square(x_dot_cv - (np.matmul(theta_cv, xi.T))).mean()
        mse_list.append(math.log10(mse_cv))
        complexity_list.append(complexity)
    plt.plot(complexity_list, mse_list)
    plt.xlabel('complexity (percentage of nonzero entries)')
    plt.ylabel('log10 of cross validation mean squared error')
    plt.savefig(os.path.join(OUTPUT_DIR, 'mse_complexity.png'))


if __name__ == '__main__':
    run()
