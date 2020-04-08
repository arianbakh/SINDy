import numpy as np
import os
import sys
import tarfile
import urllib.request


# File and Directory Settings
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')
UCI_ONLINE_URL = 'http://konect.uni-koblenz.de/downloads/tsv/opsahl-ucsocial.tar.bz2'
UCI_ONLINE_TAR_PATH = os.path.join(DATA_DIR, 'opsahl-ucsocial.tar.bz2')
UCI_ONLINE_DIR = os.path.join(DATA_DIR, 'opsahl-ucsocial')
UCI_ONLINE_TSV_PATH = os.path.join(UCI_ONLINE_DIR, 'out.opsahl-ucsocial')


# Input Data Settings
TEMPORAL_BUCKET_SIZE = 24 * 60 * 60  # in seconds  # originally 3 hours


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


def _normalize_x(x):
    return x + 1  # TODO


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
    number_of_nodes = max_id - min_id + 1
    number_of_buckets = int((last_timestamp - first_timestamp) / TEMPORAL_BUCKET_SIZE) + 1

    adjacency_matrix = np.zeros((number_of_nodes, number_of_nodes))
    x = np.zeros((number_of_buckets, number_of_nodes))
    for from_id, to_id, count, timestamp in _data_generator():
        adjacency_matrix[from_id - min_id, to_id - min_id] = 1
        bucket = int((timestamp - first_timestamp) / TEMPORAL_BUCKET_SIZE)
        x[bucket, from_id - min_id] += count

    return adjacency_matrix, _normalize_x(x)


def _get_x_dot(x):
    x_dot = (x[1:] - x[:len(x) - 1])
    return x_dot


def _calculate_mse(adjacency_matrix, x, x_dot):
    number_of_nodes = x.shape[1]
    time_frames = x.shape[0] - 1

    theta_list = []
    for i in range(number_of_nodes):
        x_i = x[:time_frames, i]
        adjacency_sum = np.sum(adjacency_matrix[:, i])
        column_list = [
            x_i ** 1.27,
            adjacency_sum * x_i ** 0.46,
        ]
        terms = []
        for j in range(number_of_nodes):
            if j != i and adjacency_matrix[j, i]:
                x_j = x[:time_frames, j]
                terms.append(
                    adjacency_matrix[j, i] * x_i ** 0.46 * x_j ** -0.54)
        if terms:
            column = np.sum(terms, axis=0)
            column_list.append(column)
        else:
            column_list.append(np.zeros(time_frames))
        theta_i = np.column_stack(column_list)
        theta_list.append(theta_i)
    theta = np.concatenate(theta_list)
    stacked_x_dot = np.concatenate([x_dot[:, node_index] for node_index in range(number_of_nodes)])
    coefficients = np.linalg.lstsq(theta, stacked_x_dot, rcond=None)[0]
    print(coefficients)  # TODO remove
    x_dot_hat = np.matmul(theta, coefficients.T)
    return np.mean((stacked_x_dot - x_dot_hat) ** 2)


def run():
    adjacency_matrix, x = _get_data_matrices()
    x_dot = _get_x_dot(x)
    mse = _calculate_mse(adjacency_matrix, x, x_dot)
    print(mse)


if __name__ == '__main__':
    run()
