"""The one-line interface to this library.

Do not import this module directly, but rather import the main ennemi module.
"""

import numpy as np
from .entropy_estimators import estimate_single_mi, estimate_conditional_mi

def estimate_mi(x : np.ndarray, y : np.ndarray, time_lag = 0, 
                k : int = 3, cond : np.ndarray = None, cond_lag : int = 0):
    """Estimate the mutual information between y and each x variable.

    Returns the estimated mutual information (in nats) for continuous
    variables. The result is a 2D array where the first index represents `x`
    rows and the second index represents the `time_lags` values.

    The time lags are applied to the `x` and `cond` arrays such that the `y`
    array stays the same every time.
    This means that `y` is cropped to `y[max_lag:N+min(min_lag, 0)]`.

    If the `cond` parameter is set, conditional mutual information is estimated.
    The `cond_lag` parameter is added to the lag for the `cond` array.
    
    If the data set contains discrete variables or many identical
    observations, this method may return incorrect results or `-inf`.
    In that case, add low-amplitude noise to the data and try again.

    The calculation is based on Kraskov et al. (2004): Estimating mutual
    information. Physical Review E 69. doi:10.1103/PhysRevE.69.066138

    Required parameters:
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

    Optional parameters:
    ---
    k : int
        The number of neighbors to consider. Default 3.
        Must be smaller than the number of observations left after cropping.
    cond : array_like
        A 1D array of observations used for conditioning.
        Must have as many observations as y.
    cond_lag : int
        Additional lag applied to the cond array. Default 0.
    """

    # The code below assumes that time_lag is an array
    if (isinstance(time_lag, int)):
        time_lag = [time_lag]
    time_lag = np.asarray(time_lag)

    # If x or y is a Python list, convert it to an ndarray
    x = np.asarray(x)
    y = np.asarray(y)

    # These are used for determining the y range to use
    min_lag = min(np.min(time_lag), np.min(time_lag+cond_lag))
    max_lag = max(np.max(time_lag), np.max(time_lag+cond_lag))

    # Validate that the lag is not too large
    if max_lag - min_lag >= y.size or max_lag >= y.size or min_lag <= -y.size:
        raise ValueError("lag is too large, no observations left")

    if x.ndim == 1:
        # If there is just one variable, the loop in the 'else' block would not work
        result = np.asarray([_lagged_mi(x, y, lag, min_lag, max_lag, k, cond, cond_lag) for lag in time_lag])

        # Force the result to be a 2D array
        result.shape = (1, len(result))
        return result
    else:
        # Go through each variable and time lag combination
        result = np.empty((len(x), len(time_lag)))
        for i in range(len(x)):
            result[i] = [_lagged_mi(x[i], y, lag, min_lag, max_lag, k, cond, cond_lag) for lag in time_lag]

        return result

def _lagged_mi(x : np.ndarray, y : np.ndarray, lag : int,
               min_lag : int, max_lag : int, k : int,
               cond : np.ndarray, cond_lag : int):
    # The x observations start from max_lag - lag
    xs = x[max_lag-lag : len(x)-lag+min(min_lag, 0)]
    # The y observations always start from max_lag
    ys = y[max_lag : len(y)+min(min_lag, 0)]

    if cond is None:
        return estimate_single_mi(xs, ys, k)
    else:
        # The cond observations have their additional lag term
        zs = cond[max_lag-(lag+cond_lag) : len(cond)-(lag+cond_lag)+min(min_lag, 0)]
        return estimate_conditional_mi(xs, ys, zs, k)

