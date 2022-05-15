import numpy as np


def chi_squared_bin_wise(y_obs: np.ndarray, y_exp: np.ndarray):
    # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm
    return np.sqrt(np.nansum(np.square((y_obs - y_exp)) / y_exp))
