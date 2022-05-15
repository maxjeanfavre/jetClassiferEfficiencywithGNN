import numpy as np


def compute_product_independent_errors(a, b, a_err, b_err):
    f = a * b

    f_err = np.sqrt(np.square(b) * np.square(a_err) + np.square(a) * np.square(b_err))

    if not np.array_equal(np.isnan(f), np.isnan(a) | np.isnan(b)):
        # places of nan values in f are not exactly the places
        # where either a or b have nan values
        raise ValueError("Placement of nan values in 'f' are not as expected")

    if not np.array_equal(
        np.isnan(f_err), np.isnan(a) | np.isnan(b) | np.isnan(a_err) | np.isnan(b_err)
    ):
        # places of nan values in f_err are not exactly the places
        # where either a, b, a_err, or b_err have nan values
        raise ValueError("Placement of nan values in 'f_err' are not as expected")

    return f, f_err


# n = 10 ** 7
# nan_values = int(0.1 * n)
#
# a = np.random.random(size=n)
# b = np.random.random(size=n)
# a_err = 0.1 * a
# b_err = 0.05 * b
#
# for arr in [a, b, a_err, b_err]:
#     m = np.random.choice(arr.size, size=nan_values, replace=False)
#     arr[m] = np.nan
#
# f, f_err = compute_product_independent_errors(a=a, b=b, a_err=a_err, b_err=b_err)
