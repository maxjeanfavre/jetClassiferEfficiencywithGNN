import numpy as np


def compute_rmse_distance(y_1: np.ndarray, y_2: np.ndarray):
    rmse = np.sqrt(np.mean(np.square(y_1 - y_2)))

    return rmse
