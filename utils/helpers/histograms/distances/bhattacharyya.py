import numpy as np


def compute_bhattacharyya_distance(y_1: np.ndarray, y_2: np.ndarray):
    res = np.sqrt(
        1 - (1.0 / np.sqrt(np.sum(y_1) * np.sum(y_2))) * np.sum(np.sqrt(y_1 * y_2))
    )

    return res
