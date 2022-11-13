import warnings

import numpy as np


def chi_squared_bin_wise(y_obs: np.ndarray, y_exp: np.ndarray):
    # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm

    # keeping the infinities from division by zero if there is an expected value of 0
    assert type(y_exp) == np.ndarray
    assert type(y_obs) == np.ndarray

    remove_idx = np.where(y_exp == 0)[0]
    y_exp = np.delete(y_exp,remove_idx)
    y_obs = np.delete(y_obs,remove_idx)
    a = np.square((y_obs - y_exp))
    #if (np.where(y_exp == 0)[0])

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=RuntimeWarning)
        b = a / y_exp

    res = np.sqrt(np.nansum(b))

    return res
