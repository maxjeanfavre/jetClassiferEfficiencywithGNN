import numpy as np


def compute_sum_independent_errors(
    vals: np.ndarray, errs: np.ndarray, ignore_nan: bool
):
    # inputs have to be np.ndarray otherwise nan values aren't taken into account as
    # desired (e.g. np.sum(a) should be nan if a contains nan values, works if a is
    # np.ndarray, doesn't work if a is a pd.Series)
    if not isinstance(vals, np.ndarray):
        raise ValueError(f"'vals' has to be np.ndarray. Got: {type(vals)}")
    if not isinstance(errs, np.ndarray):
        raise ValueError(f"'errs' has to be np.ndarray. Got: {type(vals)}")
    if ignore_nan:
        f = np.nansum(vals)
        f_err = np.sqrt(np.nansum(np.square(errs)))
    else:
        f = np.sum(vals)
        f_err = np.sqrt(np.sum(np.square(errs)))

        if np.any(np.isnan(vals)) and not np.isnan(f):
            # this should never happen by design, but just to be sure
            raise ValueError(
                "'f' wasn't nan even though there were nan values in 'vals'"
            )
        if np.any(np.isnan(errs)) and not np.isnan(f_err):
            # this should never happen by design, but just to be sure
            raise ValueError(
                "'f_err' wasn't nan even though there were nan values in 'errs'"
            )

    return f, f_err


# n = 10 ** 7
# nan_values = int(0.1 * n)
#
# vals = np.random.random(size=n)
# errs = 0.1 * vals
#
# for arr in [vals, errs]:
#     m = np.random.choice(arr.size, size=nan_values, replace=False)
#     arr[m] = np.nan
#
# f, f_err = compute_sum_independent_errors(vals=vals, errs=errs, ignore_nan=False)
#
# assert np.isnan(f)
# assert np.isnan(f_err)
#
# f, f_err = compute_sum_independent_errors(vals=vals, errs=errs, ignore_nan=True)
