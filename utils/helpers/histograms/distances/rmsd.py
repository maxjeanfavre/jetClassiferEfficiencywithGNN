import numpy as np


def compute_rmsd_distance(y_1: np.ndarray, y_2: np.ndarray):
    rmsd = np.sqrt(np.mean(np.square(y_1 - y_2)))

    return rmsd
