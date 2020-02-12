import numpy as np
import random
import sys


# TODO lambda needs to be calculated
# TODO save final differential equation (latex) and graph structure to pdf
# TODO one regression to rule them all


NUMBER_OF_NODES = 2  # TODO
DELTA_T = 0.01
TIME_FRAMES = 100
SINDY_ITERATIONS = 10
LAMBDA = 0.1
POWERS = np.arange(0.5, 2.5, 0.5).tolist()


def _get_x():
    x = np.zeros((TIME_FRAMES + 1, NUMBER_OF_NODES))
    a = np.zeros((NUMBER_OF_NODES, NUMBER_OF_NODES))
    for i in range(NUMBER_OF_NODES):
        for j in range(NUMBER_OF_NODES):
            if i != j:
                # a[i, j] = random.random()
                a[i, j] = 1
    x[0] = np.array([random.random() for i in range(NUMBER_OF_NODES)])
    for i in range(1, TIME_FRAMES + 1):
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
    for i in range(TIME_FRAMES):
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


def _sindy(x_dot, theta):
    xi = np.zeros((NUMBER_OF_NODES, theta.shape[1]))
    total_residuals = 0
    for i in range(NUMBER_OF_NODES):
        ith_derivative = x_dot[:, i]
        ith_xi = np.linalg.lstsq(theta, ith_derivative, rcond=None)[0]
        residuals = 0
        for j in range(SINDY_ITERATIONS):
            small_indices = np.flatnonzero(np.absolute(ith_xi) < LAMBDA)
            big_indices = np.flatnonzero(np.absolute(ith_xi) >= LAMBDA)
            ith_xi[small_indices] = 0
            least_squares = np.linalg.lstsq(theta[:, big_indices], ith_derivative, rcond=None)
            ith_xi[big_indices] = least_squares[0]
            residuals = sum(least_squares[3])
        total_residuals += residuals
        xi[i] = ith_xi
    print(total_residuals)
    return xi


def run():
    x = _get_x()
    x_dot = _get_x_dot(x)
    theta = _get_theta(x)
    xi = _sindy(x_dot, theta)
    print(xi)


if __name__ == '__main__':
    run()
