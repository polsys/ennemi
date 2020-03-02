"""The one-line interface to this library.

Do not import this module directly, but rather import the main ennemi module.
"""

import numpy as np
from .entropy_estimators import estimate_single_mi

def estimate_mi(x : np.ndarray, y : np.ndarray, time_lag = 0, 
                k : int = 3):
    """Estimate the mutual information between y and each x variable.

    Returns the estimated mutual information (in nats) for continuous
    variables. The result is a 2D array where the first index represents `x`
    rows and the second index represents the `time_lags` values.

    The time lags are applied to the `x` array such that the `y` array stays the
    same every time.
    This means that `y` is cropped to `y[max_lag:N+min(min_lag, 0)]`.
    
    If the data set contains discrete variables or many identical
    observations, this method may return incorrect results or `-inf`.
    In that case, add low-amplitude noise to the data and try again.

    The calculation is based on Kraskov et al. (2004): Estimating mutual
    information. Physical Review E 69. doi:10.1103/PhysRevE.69.066138

    Parameters:
    ---
    x : array_like
        A 1D or 2D array where the rows are one or more variables and the
        columns are observations. The number of columns must be the same as in y.
    y : array_like
        A 1D array of observations.
    time_lag : int or array_like
        A time lag or 1D array of time lags to apply to x. Default 0.
        The values may be any integers with magnitude
        less than the number of observations.
    k : int
        The number of neighbors to consider. Default 3.
        Must be smaller than the number of observations left after cropping.
    """

    # These are used for determining the y range to use
    min_lag = np.min(time_lag)
    max_lag = np.max(time_lag)

    # The code below assumes that time_lag is an array
    if (isinstance(time_lag, int)):
        time_lag = [time_lag]

    # If x or y is a Python list, convert it to an ndarray
    x = np.asarray(x)
    y = np.asarray(y)

    # Validate that the lag is not too large
    if max_lag - min_lag >= y.size or max_lag >= y.size or min_lag <= -y.size:
        raise ValueError("lag is too large, no observations left")

    if x.ndim == 1:
        # If there is just one variable, the loop in the 'else' block would not work
        result = np.asarray([_lagged_mi(x, y, lag, min_lag, max_lag, k) for lag in time_lag])

        # Force the result to be a 2D array
        result.shape = (1, len(result))
        return result
    else:
        # Go through each variable and time lag combination
        result = np.empty((len(x), len(time_lag)))
        for i in range(len(x)):
            result[i] = [_lagged_mi(x[i], y, lag, min_lag, max_lag, k) for lag in time_lag]

        return result

def _lagged_mi(x : np.ndarray, y : np.ndarray, lag : int,
               min_lag : int, max_lag : int, k : int):
    # The x observations start from max_lag - lag
    xs = x[max_lag-lag : len(x)-lag+min(min_lag, 0)]
    # The y observations always start from max_lag
    ys = y[max_lag : len(y)+min(min_lag, 0)]

    return estimate_single_mi(xs, ys, k)
