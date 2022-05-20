import warnings

import numpy as np


def chi_squared_bin_wise(y_obs: np.ndarray, y_exp: np.ndarray):
    # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm

    # keeping the infinities from division by zero if there is an expected value of 0
    a = np.square((y_obs - y_exp))

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=RuntimeWarning)
        b = a / y_exp

    res = np.sqrt(np.nansum(b))

    return res
