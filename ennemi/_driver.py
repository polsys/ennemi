"""The one-line interface to this library.

Do not import this module directly, but rather import the main ennemi module.
"""

import concurrent.futures
import itertools
import numpy as np
from ._entropy_estimators import _estimate_single_mi, _estimate_conditional_mi

def estimate_mi(y : np.ndarray, x : np.ndarray, time_lag = 0, 
                k : int = 3, cond : np.ndarray = None, cond_lag : int = 0,
                parallel : str = None):
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
    y : array_like
        A 1D array of observations.
    x : array_like
        A 1D or 2D array where the rows are one or more variables and the
        columns are observations. The number of columns must be the same as in y.
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
    parallel : str or None
        Whether to run the estimation in multiple processes. If None (the default),
        a heuristic will be used for the decision. If "always", each
        variable / time lag combination will be run in a separate subprocess,
        with as many concurrent processes as there are processors.
        If "disable", the combinations are estimated sequentially in the current process.
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
        nvar = 1
    else:
        nvar = len(x)

    # Create a list of all variable, time lag combinations
    # The params map contains tuples for simpler passing into subprocess
    indices = list(itertools.product(range(nvar), range(len(time_lag))))
    if x.ndim == 1:
        params = map(lambda lag: (x, y, lag, max_lag, min_lag, k, cond, cond_lag), time_lag)
    else:
        params = map(lambda i: (x[i[0],:], y, time_lag[i[1]], max_lag, min_lag, k, cond, cond_lag), indices)

    # If there is benefit in doing so, and the user has not overridden the
    # heuristic, execute the estimation in multiple parallel processes
    # TODO: In a many variables/lags, small N case, it may make sense to
    #       use multiple processes, but batch the tasks
    if parallel == "always":
        use_parallel = True
    elif parallel == "disable":
        use_parallel = False
    elif parallel is not None:
        raise ValueError("unrecognized value for parallel argument")
    else:
        # As parallel is None, use a heuristic
        use_parallel = len(indices) > 1 and len(y) > 200

    if use_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            conc_result = executor.map(_do_estimate, params)
    else:
        conc_result = map(_do_estimate, params)
    
    # Collect the results to a 2D array
    result = np.empty((nvar, len(time_lag)))
    for index, res in zip(indices, conc_result):
        result[index] = res
        
    return result


def _do_estimate(param_tuple):
    # A helper for unpacking the param tuple (maybe unnecessary?)
    x, y, lag, max_lag, min_lag, k, cond, cond_lag = param_tuple
    return _lagged_mi(x, y, lag, min_lag, max_lag, k, cond, cond_lag)


def _lagged_mi(x : np.ndarray, y : np.ndarray, lag : int,
               min_lag : int, max_lag : int, k : int,
               cond : np.ndarray, cond_lag : int):
    # The x observations start from max_lag - lag
    xs = x[max_lag-lag : len(x)-lag+min(min_lag, 0)]
    # The y observations always start from max_lag
    ys = y[max_lag : len(y)+min(min_lag, 0)]

    if cond is None:
        return _estimate_single_mi(xs, ys, k)
    else:
        # The cond observations have their additional lag term
        zs = cond[max_lag-(lag+cond_lag) : len(cond)-(lag+cond_lag)+min(min_lag, 0)]
        return _estimate_conditional_mi(xs, ys, zs, k)

