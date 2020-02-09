import numpy as np


DELTA_T = 0.001
SIGMA = 10
RHO = 28
BETA = 8 / 3
TIME_FRAMES = 100
SINDY_ITERATIONS = 10
LAMBDA = 0.001


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


def _get_x_dot(x):
    x_dot = (x[1:] - x[:len(x) - 1]) / DELTA_T
    return x_dot


def _get_theta(x):
    theta = np.zeros((TIME_FRAMES, 7))
    for i in range(TIME_FRAMES):
        theta[i] = np.array([
            1,
            x[i][0],
            x[i][1],
            x[i][2],
            x[i][0] * x[i][1],
            x[i][1] * x[i][2],
            x[i][0] * x[i][2],
        ])
    return theta


def _sindy(x_dot, theta):
    xi = np.zeros((3, 7))
    for i in range(3):
        ith_derivative = x_dot[:, i]
        ith_xi = np.linalg.lstsq(theta, ith_derivative, rcond=None)[0]
        for j in range(SINDY_ITERATIONS):
            small_indices = np.flatnonzero(np.absolute(ith_xi) < LAMBDA)
            big_indices = np.flatnonzero(np.absolute(ith_xi) >= LAMBDA)
            ith_xi[small_indices] = 0
            ith_xi[big_indices] = np.linalg.lstsq(theta[:, big_indices], ith_derivative, rcond=None)[0]
        xi[i] = ith_xi
    return xi


def run():
    initial_vector = np.array([1, 1, 1])
    x = _get_x(initial_vector)
    x_dot = _get_x_dot(x)
    theta = _get_theta(x)
    xi = _sindy(x_dot, theta)
    print(xi)


if __name__ == '__main__':
    run()
