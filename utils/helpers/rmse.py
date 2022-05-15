import numpy as np


def compute_rmse(y_true, y_pred):
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))

    return rmse
