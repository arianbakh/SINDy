import itertools
import math
import numpy as np
import sys


DELTA_T = 0.001
SIGMA = 10
RHO = 28
BETA = 8 / 3
TIME_FRAMES = 1000
SINDY_ITERATIONS = 10
CANDIDATE_LAMBDAS = [2 ** -i for i in range(101)]
CV_PERCENTAGE = 0.2
MAX_POWER = 2


def _get_x(initial_vector):
    x = np.zeros((TIME_FRAMES + 1, 3))
    x[0] = initial_vector
    for i in range(1, TIME_FRAMES + 1):
        prev_x = x[i - 1][0]
        prev_y = x[i - 1][1]
        prev_z = x[i - 1][2]
        x[i] = [
            prev_x + DELTA_T * SIGMA * (prev_y - prev_x),
            prev_y + DELTA_T * (prev_x * (RHO - prev_z) - prev_y),
            prev_z + DELTA_T * (prev_x * prev_y - BETA * prev_z),
        ]
    return x


def _normalize_x(x):
    normalized_columns = []
    for column_index in range(x.shape[1]):
        column = x[:, column_index]
        std = max(10 ** -9, np.std(column))  # to avoid division by zero
        normalized_column = (column - np.mean(column)) / std
        normalized_columns.append(normalized_column)
    normalized_x = np.column_stack(normalized_columns)
    normalized_x -= np.min(normalized_x)  # change minimum to zero
    normalized_x += 1  # change minimum to one
    return normalized_x


def _get_x_dot(x):
    x_dot = (x[1:] - x[:len(x) - 1]) / DELTA_T
    return x_dot


def _get_theta(x):
    time_frames = x.shape[0] - 1
    theta_columns = [np.ones(time_frames)]
    vectors = [x[:time_frames, i] for i in range(x.shape[1])]
    for subset in itertools.combinations(vectors, 1):
        for power in range(1, MAX_POWER + 1):
            theta_columns.append(subset[0] ** power)
    for subset in itertools.combinations(vectors, 2):
        for first_power in range(1, MAX_POWER + 1):
            for second_power in range(1, MAX_POWER + 1):
                theta_columns.append((subset[0] ** first_power) * (subset[1] ** second_power))
    for subset in itertools.combinations(vectors, 3):
        theta_columns.append(subset[0] * subset[1] * subset[2])
    theta = np.column_stack(theta_columns)
    return theta


def _sindy(x_dot, theta, candidate_lambda):
    xi = np.zeros((x_dot.shape[1], theta.shape[1]))
    for i in range(x_dot.shape[1]):
        ith_derivative = x_dot[:, i]
        ith_xi = np.linalg.lstsq(theta, ith_derivative, rcond=None)[0]
        for j in range(SINDY_ITERATIONS):
            small_indices = np.flatnonzero(np.absolute(ith_xi) < candidate_lambda)
            big_indices = np.flatnonzero(np.absolute(ith_xi) >= candidate_lambda)
            ith_xi[small_indices] = 0
            ith_xi[big_indices] = np.linalg.lstsq(theta[:, big_indices], ith_derivative, rcond=None)[0]
        xi[i] = ith_xi
    return xi


def _optimum_sindy(x_dot, theta, x_dot_cv, theta_cv):
    least_cost = sys.maxsize
    best_xi = None
    best_mse = -1
    best_complexity = -1
    for candidate_lambda in CANDIDATE_LAMBDAS:
        xi = _sindy(x_dot, theta, candidate_lambda)
        complexity = np.count_nonzero(xi)
        x_dot_cv_hat = np.matmul(theta_cv, xi.T)
        mse = np.square(x_dot_cv - x_dot_cv_hat).mean()
        if complexity:  # zero would mean no statements
            cost = mse * complexity
            if cost < least_cost:
                least_cost = cost
                best_xi = xi
                best_mse = mse
                best_complexity = complexity
    print('best log10(MSE):', math.log10(best_mse))
    print('best complexity:', best_complexity)
    return best_xi


def run():
    initial_vector = np.array([1, 1, 1])
    x = _get_x(initial_vector)
    x = _normalize_x(x)  # TODO uncomment
    entire_x_dot = _get_x_dot(x)
    entire_theta = _get_theta(x)
    cv_index = int(entire_x_dot.shape[0] * (1 - CV_PERCENTAGE))
    x_dot = entire_x_dot[:cv_index]
    x_dot_cv = entire_x_dot[cv_index:]
    theta = entire_theta[:cv_index]
    theta_cv = entire_theta[cv_index:]
    xi = _optimum_sindy(x_dot ** 2, theta, x_dot_cv ** 2, theta_cv)


if __name__ == '__main__':
    run()
