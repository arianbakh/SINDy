import itertools
import math
import numpy as np
import sys

from scipy.stats import pearsonr


DELTA_T = 0.001
SIGMA = 10
RHO = 28
BETA = 8 / 3
TIME_FRAMES = 10000
SINDY_ITERATIONS = 10
CV_PERCENTAGE = 0.2
MAX_POWER = 4
LAMBDA_RANGE = [0.0001, 0.1]
LAMBDA_STEP = 2


# Calculated Settings
CANDIDATE_LAMBDAS = [
    LAMBDA_STEP ** i for i in range(
        int(math.log(abs(LAMBDA_RANGE[0])) / math.log(LAMBDA_STEP)),
        int(math.log(abs(LAMBDA_RANGE[1])) / math.log(LAMBDA_STEP))
    )
]


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
    theta = np.column_stack(theta_columns)
    return theta


def _single_node_sindy(x_dot_i, theta, candidate_lambda):
    xi_i = np.linalg.lstsq(theta, x_dot_i, rcond=None)[0]
    for j in range(SINDY_ITERATIONS):
        small_indices = np.flatnonzero(np.absolute(xi_i) < candidate_lambda)
        big_indices = np.flatnonzero(np.absolute(xi_i) >= candidate_lambda)
        xi_i[small_indices] = 0
        xi_i[big_indices] = np.linalg.lstsq(theta[:, big_indices], x_dot_i, rcond=None)[0]
    return xi_i


def _optimum_sindy(x_dot, theta):
    cv_index = int(x_dot.shape[0] * (1 - CV_PERCENTAGE))
    x_dot_train = x_dot[:cv_index]
    x_dot_cv = x_dot[cv_index:]
    theta_train = theta[:cv_index]
    theta_cv = theta[cv_index:]

    xi = np.zeros((x_dot_train.shape[1], theta_train.shape[1]))
    for i in range(x_dot_train.shape[1]):
        least_cost = sys.maxsize
        best_xi_i = None
        best_mse = sys.maxsize
        x_dot_i = x_dot_train[:, i]
        x_dot_cv_i = x_dot_cv[:, i]
        for candidate_lambda in CANDIDATE_LAMBDAS:
            xi_i = _single_node_sindy(x_dot_i, theta_train, candidate_lambda)
            complexity = math.log(1 + np.count_nonzero(xi_i))
            x_dot_hat_i = np.matmul(theta_cv, xi_i.T)
            mse_cv = np.square(x_dot_cv_i - x_dot_hat_i).mean()
            if complexity:  # zero would mean no statements
                cost = mse_cv * complexity
                if cost < least_cost:
                    least_cost = cost
                    best_mse = mse_cv
                    best_xi_i = xi_i
        xi[i] = best_xi_i
        print('Node', i, 'best log10(MSE):', math.log10(best_mse))
    print()  # newline
    return xi


def _get_pearson_correlation_coefficient(x_dot, theta):
    pcc = np.zeros((x_dot.shape[1], theta.shape[1]))
    for i in range(x_dot.shape[1]):
        for j in range(theta.shape[1]):
            pcc[i, j] = pearsonr(x_dot[:, i], theta[:, j])[0]
    return np.nan_to_num(pcc)


def run():
    initial_vector = np.array([1, 1, 1])
    x = _get_x(initial_vector)
    # x = _normalize_x(x)  # TODO
    entire_x_dot = _get_x_dot(x)
    entire_theta = _get_theta(x)
    xi = _optimum_sindy(entire_x_dot, entire_theta)
    print(xi)
    pcc = _get_pearson_correlation_coefficient(entire_x_dot, entire_theta)
    print(pcc)


if __name__ == '__main__':
    run()
