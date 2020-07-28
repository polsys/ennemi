# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

"""The one-line interface to this library.

Do not import this module directly, but rather import the main ennemi module.
"""

import concurrent.futures
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union
import itertools
import math
import numpy as np
from os import cpu_count
import sys
from ._entropy_estimators import _estimate_single_mi, _estimate_conditional_mi,\
    _estimate_semidiscrete_mi, _estimate_conditional_semidiscrete_mi, _estimate_single_entropy

ArrayLike = Union[List[float], List[Tuple[float, ...]], np.ndarray]
GenArrayLike = TypeVar("GenArrayLike", List[float], List[Tuple[float, ...]], np.ndarray)
T = TypeVar("T")

def normalize_mi(mi: GenArrayLike) -> GenArrayLike:
    """Normalize mutual information values to the unit interval.

    The return value matches the correlation coefficient between two Gaussian
    random variables with unit variance. This coefficient is preserved by all
    monotonic transformations, including scaling. The value is positive regardless
    of the sign of the correlation.

    Negative values are kept as-is. This is because mutual information is always
    non-negative, but `estimate_mi` may produce negative values.

    Parameters:
    ---
    mi : array_like or float
        One or more mutual information values (in nats).
        If this is a Pandas `DataFrame` or `Series`, the columns and indices
        are preserved.
    """

    # If the parameter is a pandas type, preserve the columns and indices
    if "pandas" in sys.modules:
        import pandas
        if isinstance(mi, (pandas.DataFrame, pandas.Series)):
            return mi.applymap(_normalize)
    
    return np.vectorize(_normalize, otypes=[np.float])(mi)

def _normalize(mi: np.float) -> np.float:
    if mi <= 0.0:
        return mi
    else:
        return np.sqrt(1 - np.exp(-2 * mi))


def estimate_entropy(x: ArrayLike,
    *, k: int = 3,
    multidim: bool = False,
    mask: Optional[ArrayLike] = None,
    cond: Optional[ArrayLike] = None,
    drop_nan: bool = False) -> np.ndarray:
    """Estimate the entropy of one or more continuous random variables.

    Returns the estimated entropy in nats. If `x` is two-dimensional, each
    marginal variable is estimated separately by default. If the `multidim`
    parameter is set to True, the array is interpreted as a single n-dimensional
    random variable. If `x` is a pandas `DataFrame` or `Series`, the result is
    a `DataFrame`.

    If the `mask` parameter is set, only those `x` observations with the
    matching mask element set to `True` are used for estimation.

    The calculation is described in Kraskov et al. (2004): Estimating mutual
    information. Physical Review E 69. doi:10.1103/PhysRevE.69.066138

    Positional or keyword parameters:
    ---
    x : array_like
        A 1D or 2D array of observations. The interpretation of columns
        depends on the `multidim` parameter.

    Optional keyword parameters:
    ---
    k : int, default 3
        The number of neighbors to consider.
    multidim : bool
        If False (the default), each column of `x` is considered a separate variable.
        If True, the (n x m) array is considered a single m-dimensional variable.
    mask : array_like or None
        If specified, an array of booleans that gives the input elements to use for
        estimation. Use this to exclude some observations from consideration.
        Currently, the mask must be one-dimensional.
    cond : array_like or None
        Optional 1D or 2D array of observations used for conditioning.
        All variables in a 2D array are used together.
        The calculation uses the chain rule H(X|Y) = H(X,Y) - H(Y) without
        any correction for potential estimation bias.
    drop_nan : bool, default False
        If True, all NaN (not a number) values are masked out.
    """

    x_arr = np.asarray(x)

    # Validate the parameters
    # Post-masking validation is done just before going to the algorithm
    if mask is not None:
        mask = np.asarray(mask)
        _validate_mask(mask, x_arr.shape[0])
    if not 1 <= x_arr.ndim <= 2:
        raise ValueError("x must be one- or two-dimensional")
    _validate_k_type(k)

    if cond is None:
        result = _estimate_entropy(x_arr, k, multidim, mask, drop_nan)
    else:
        cond_arr = np.asarray(cond)
        _validate_cond(cond_arr, x_arr.shape[0])
        result = _estimate_conditional_entropy(x_arr, cond_arr, k, multidim, mask, drop_nan)

    # If the original x array was a pandas data type, return a DataFrame
    # As an exception, if multidim=True, we still return a NumPy scalar
    if not multidim and "pandas" in sys.modules:
        import pandas
        if isinstance(x, pandas.DataFrame):
            return pandas.DataFrame(np.atleast_2d(result), columns=x.columns)
        elif isinstance(x, pandas.Series):
            return pandas.DataFrame(np.atleast_2d(result), columns=[x.name])
    return result


def _estimate_entropy(x: np.ndarray, k: int, multidim: bool,
        mask: Optional[np.ndarray], drop_nan: bool) -> np.ndarray:
    """Strongly typed estimate_entropy()."""

    if multidim or x.ndim == 1:
        x = _mask_and_validate_entropy(x, mask, drop_nan, k)
        return np.asarray(_estimate_single_entropy(x, k))
    else:
        nvar = x.shape[1]
        result = np.empty(nvar)
        for i in range(nvar):
            xs = _mask_and_validate_entropy(x[:,i], mask, drop_nan, k)
            result[i] = _estimate_single_entropy(xs, k)
        return result

def _mask_and_validate_entropy(x: np.ndarray, mask: Optional[np.ndarray],
        drop_nan: bool, k: int) -> np.ndarray:
    # Apply the mask and drop NaNs
    # TODO: Support 2D masks (https://github.com/polsys/ennemi/issues/37)
    if mask is not None:
        x = x[mask]

    if drop_nan and x.ndim > 1:
        x = x[~np.max(np.isnan(x), axis=1)]
    elif drop_nan:
        x = x[~np.isnan(x)]

    # Validate the x array
    if k >= x.shape[0]:
        raise ValueError("k must be smaller than number of observations (after lag and mask)")
    if np.any(np.isnan(x)):
        raise ValueError("input contains NaNs (after applying the mask), pass drop_nan=True to ignore")

    return x

def _estimate_conditional_entropy(x: np.ndarray, cond: np.ndarray, k: int, multidim: bool,
        mask: Optional[np.ndarray], drop_nan: bool) -> np.ndarray:
    """Conditional entropy by the chain rule: H(X|Y) = H(X,Y) - H(Y)."""

    # Estimate the entropy of cond by the method above (multidim=True)
    marginal = _estimate_entropy(cond, k, True, mask, drop_nan)

    # The joint entropy depends on multidim and number of dimensions:
    # In the latter case, the joint entropy is calculated for each x variable
    if multidim or x.ndim == 1:
        xs = _mask_and_validate_entropy(np.column_stack((x, cond)), mask, drop_nan, k)
        return np.asarray(_estimate_single_entropy(xs, k) - marginal)
    else:
        nvar = x.shape[1]
        joint = np.empty(nvar) # type: np.ndarray
        for i in range(nvar):
            xs = _mask_and_validate_entropy(np.column_stack((x[:,i], cond)), mask, drop_nan, k)
            joint[i] = _estimate_single_entropy(xs, k)
        return joint - marginal


def estimate_mi(y: ArrayLike, x: ArrayLike,
                lag: Union[Sequence[int], np.ndarray, int] = 0,
                *, k: int = 3,
                cond: Optional[ArrayLike] = None,
                cond_lag: Union[Sequence[int], np.ndarray, int] = 0,
                mask: Optional[ArrayLike] = None,
                discrete_y: bool = False,
                preprocess: bool = True,
                drop_nan: bool = False,
                normalize: bool = False,
                max_threads: Optional[int] = None,
                callback: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """Estimate the mutual information between y and each x variable.
 
    Returns the estimated mutual information (in nats) for continuous
    variables. The result is a 2D `ndarray` where the first index represents `x`
    rows and the second index represents the `lag` values. If `x` is a pandas
    `DataFrame` or `Series`, the result is a `DataFrame`.

    The time lag is interpreted as `y(t) ~ x(t - lag) | z(t - cond_lag)`.
    The time lags are applied to the `x` and `cond` arrays such that the `y`
    array stays the same every time.
    This means that `y` is cropped to `y[max(max_lag,0) : N+min(min_lag,0)]`.

    If the `cond` parameter is set, conditional mutual information is estimated.
    The `cond_lag` parameter specifies the lag for the `cond` array, separately
    from the `x` lag.

    If the `mask` parameter is set, only those `y` observations with the
    matching mask element set to `True` are used for estimation.
    
    For accurate results, the variables should be transformed to roughly
    symmetrical distributions. If `preprocess==True`, the variables are
    automatically scaled to unit variance.

    The calculation is based on "Kraskov et al. (2004): Estimating mutual
    information. Physical Review E 69. doi:10.1103/PhysRevE.69.066138" and
    derivatives by (Frenzel and Pompe 2007) and (Ross 2014).

    Positional or keyword parameters:
    ---
    y : array_like
        A 1D array of observations. If `discrete_y` is True, the values may be
        of any type. Otherwise the values must be numeric.
    x : array_like
        A 1D or 2D array where the columns are one or more variables and the
        rows are observations. The number of rows must be the same as in y.
    lag : int or array_like, default 0
        A time lag or 1D array of time lags to apply.

    Optional keyword parameters:
    ---
    k : int, default 3
        The number of neighbors to consider.
    cond : array_like or None
        Optional 1D or 2D array of observations used for conditioning.
        Must have as many observations as y.
        All variables in a 2D array are used together.
    cond_lag : int or array_like, default 0
        Lag applied to the cond array. Must be broadcastable to the size of `lag`.
    mask : array_like or None
        If specified, an array of booleans that gives the `y` elements to use for
        estimation. Use this to exclude some observations from consideration
        while preserving the time series structure of the data. Elements of
        `x` and `cond` are masked with the lags applied.
    discrete_y : bool, default False
        If True, the `y` variable is interpreted as a discrete variable. The `x`
        variables are still continuous. The `y` values may be non-numeric.
        Default False.
    preprocess : bool, default True
        If True (the default), the variables are scaled to unit variance and
        added with low-amplitude noise. The noise uses a fixed random seed.
    drop_nan : bool, default False
        If True, all NaN (not a number) values are masked out.
    normalize : bool, default False
        If True, the results will be normalized to correlation coefficient scale.
        Same as calling `normalize_mi` on the results.
    max_threads : int or None
        The maximum number of threads to use for estimation.
        If None (the default), the number of CPU cores is used.
    callback : method or None
        A method to call when each estimation task is completed. The method
        must take two integer parameters: `x` variable index and lag value.
        This method should be very short. Because Python code is not executed
        concurrently, the callback may slow down other estimation tasks.
    """

    # Convert parameters to consistent types
    lag_arr = np.atleast_1d(lag)
    cond_lag_arr = np.broadcast_to(cond_lag, lag_arr.shape)
    
    # Keep the original x parameter around for the Pandas data frame check
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    if cond is not None: cond_arr = np.asarray(cond)
    else: cond_arr = None
    if mask is not None: mask_arr = np.asarray(mask)
    else: mask_arr = None

    # Check the parameters and run the estimation
    result = _estimate_mi(y_arr, x_arr, lag_arr, k, cond_arr,
        cond_lag_arr, mask_arr, discrete_y, preprocess, drop_nan, max_threads, callback)

    # Normalize if requested
    if normalize:
        result = normalize_mi(result)

    # If the input was a pandas data frame, set the column names
    if "pandas" in sys.modules:
        import pandas
        if isinstance(x, pandas.DataFrame):
            return pandas.DataFrame(result, index=lag_arr, columns=x.columns)
        elif isinstance(x, pandas.Series):
            return pandas.DataFrame(result, index=lag_arr, columns=[x.name])
    return result

def _estimate_mi(y: np.ndarray, x: np.ndarray, lag: np.ndarray, k: int,
        cond: Optional[np.ndarray], cond_lag: np.ndarray,
        mask: Optional[np.ndarray], discrete_y: bool, preprocess: bool, drop_nan: bool,
        max_threads: Optional[int],
        callback: Optional[Callable[[int, int], None]]) -> np.ndarray:
    """This method is strongly typed, estimate_mi() does necessary conversion."""

    _check_parameters(x, y, k, cond, mask)

    # These are used for determining the y range to use
    min_lag = min(np.min(lag), np.min(cond_lag))
    max_lag = max(np.max(lag), np.max(cond_lag))

    # Validate that the lag is not too large
    if max_lag - min_lag >= y.size or max_lag >= y.size or min_lag <= -y.size:
        raise ValueError("lag is too large, no observations left")
    
    if x.ndim == 1:
        nvar = 1
    else:
        _, nvar = x.shape

    # Create a list of all variable, time lag combinations
    # The params map contains tuples for simpler passing into worker function
    indices = list(itertools.product(range(len(lag)), range(nvar)))
    if x.ndim == 1:
        params = list(map(lambda i: (x, y, lag[i[0]], max_lag, min_lag, k,
            mask, cond, cond_lag[i[0]], discrete_y, preprocess, drop_nan), indices))
    else:
        params = list(map(lambda i: (x[:,i[1]], y, lag[i[0]], max_lag, min_lag, k,
            mask, cond, cond_lag[i[0]], discrete_y, preprocess, drop_nan), indices))

    # Run the estimation, possibly in parallel
    def wrapped_callback(i: int) -> None:
        if callback is not None:
            lag_index, var_index = indices[i]
            callback(var_index, lag[lag_index])
    
    time_estimate = _get_mi_time_estimate(len(y), cond, k)
    conc_result = _map_maybe_parallel(_lagged_mi, params, max_threads, time_estimate, wrapped_callback)
    
    # Collect the results to a 2D array
    result = np.empty((len(lag), nvar))
    for index, res in zip(indices, conc_result):
        result[index] = res
    return result


def _check_parameters(x: np.ndarray, y: Optional[np.ndarray], k: int,
        cond: Optional[np.ndarray], mask: Optional[np.ndarray]) -> None:
    """Does most of parameter checking, but some is still left to _lagged_mi."""
    _validate_k_type(k)

    # Validate the array shapes and lengths
    if not 1 <= len(x.shape) <= 2:
        raise ValueError("x must be one- or two-dimensional")
    if y is not None:
        if len(y.shape) > 1:
            raise ValueError("y must be one-dimensional")
        if (x.shape[0] != y.shape[0]):
            raise ValueError("x and y must have same length")

    # Validate the mask and condition
    if mask is not None: _validate_mask(mask, x.shape[0])
    if cond is not None: _validate_cond(cond, x.shape[0])

def _validate_k_type(k: int) -> None:
    if not isinstance(k, int):
        raise TypeError("k must be int")
    if k <= 0:
        raise ValueError("k must be greater than zero")

def _validate_mask(mask: np.ndarray, input_len: int) -> None:
    if len(mask.shape) > 1:
        raise ValueError("mask must be one-dimensional")
    if len(mask) != input_len:
        raise ValueError("mask length does not match input length")
    if mask.dtype != np.bool:
        raise TypeError("mask must contain only booleans")

def _validate_cond(cond: np.ndarray, input_len: int) -> None:
    if not 1 <= cond.ndim <= 2:
        raise ValueError("cond must be one- or two-dimensional")
    if input_len != len(cond):
        raise ValueError("x and cond must have same length")


def pairwise_mi(data: ArrayLike,
    *, k: int = 3,
    cond: Optional[ArrayLike] = None,
    mask: Optional[ArrayLike] = None,
    preprocess: bool = True,
    drop_nan: bool = False,
    normalize: bool = False,
    max_threads: Optional[int] = None,
    callback: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """Estimate the pairwise MI between each variable.

    Returns a matrix where the (i,j)'th element is the mutual information
    between the i'th and j'th columns in the data. The values are in nats or
    in the normalized scale depending on the `normalize` parameter. The diagonal
    contains NaNs (for better visualization, as the auto-MI should be infinite).

    Positional or keyword parameters:
    ---
    data : array_like
        A 2D array where the columns are variables and rows are observations.

    Optional keyword parameters:
    ---
    k : int, default 3
        The number of neighbors to consider.
    cond : array_like or None
        Optional 1D or 2D array of observations used for conditioning.
        Must have as many observations as the data.
        All variables in a 2D array are used together.
    mask : array_like or None
        If specified, an array of booleans that gives the data elements to use for
        estimation. Use this to exclude some observations from consideration.
    preprocess : bool, default True
        If True (the default), the variables are scaled to unit variance and
        added with low-amplitude noise. The noise uses a fixed random seed.
    drop_nan : bool, default False
        If True, all NaN (not a number) values are masked out.
    normalize: bool, default False
        If True, the MI values will be normalized to correlation coefficient scale.
    max_threads : int or None
        The maximum number of threads to use for estimation.
        If None (the default), the number of CPU cores is used.
    callback : method or None
        A method to call when each estimation task is completed. The method
        must take two integer parameters, representing the variable indices.
        This method should be very short. Because Python code is not executed
        concurrently, the callback may slow down other estimation tasks.
    """
    data_arr = np.asarray(data)
    if cond is not None: cond_arr = np.asarray(cond)
    else: cond_arr = None
    if mask is not None: mask_arr = np.asarray(mask)
    else: mask_arr = None

    # If there is just one variable, return the trivial result
    if data_arr.ndim == 1 or data_arr.shape[1] == 1:
        return np.full((1,1), np.nan)
    
    result = _pairwise_mi(data_arr, k, cond_arr, preprocess, drop_nan, mask_arr, max_threads, callback)

    # Normalize if asked for
    if normalize:
        result = normalize_mi(result)

    # If data was a pandas DataFrame, return a DataFrame with matching names
    if "pandas" in sys.modules:
        import pandas
        if isinstance(data, pandas.DataFrame):
            return pandas.DataFrame(result, index=data.columns, columns=data.columns)
    return result


def _pairwise_mi(data: np.ndarray, k: int, cond: Optional[np.ndarray], preprocess: bool,
    drop_nan: bool, mask: Optional[np.ndarray], max_threads: Optional[int],
    callback: Optional[Callable[[int, int], None]]) -> np.ndarray:
    """Strongly typed pairwise MI. The data array is at least 2D."""

    _check_parameters(data, None, k, cond, mask)
    nobs, nvar = data.shape

    # Create a list of variable pairs
    # By symmetry, it suffices to consider a triangular matrix
    indices = []
    params = []
    for i in range(nvar):
        for j in range(i+1, nvar):
            indices.append((i, j))
            params.append((data[:,i], data[:,j], 0, 0, 0, k, mask, cond, 0, False, preprocess, drop_nan))

    # Run the MI estimation for each pair, possibly in parallel
    def wrapped_callback(i: int) -> None:
        if callback is not None:
            callback(*indices[i])

    time_estimate = _get_mi_time_estimate(nobs, cond, k)
    conc_result = _map_maybe_parallel(_lagged_mi, params, max_threads, time_estimate, wrapped_callback)

    # Collect the results, creating a symmetric matrix now
    result = np.full((nvar, nvar), np.nan)
    for (i,j), res in zip(indices, conc_result):
        result[i,j] = res
        result[j,i] = res
    
    return result


def _get_mi_time_estimate(n: int, cond: Optional[np.ndarray], k: int) -> float:
    if cond is None:
        n_cond = 0
    elif cond.ndim == 1:
        n_cond = 1
    else:
        n_cond = cond.shape[1]

    # These are determined pretty experimentally on a laptop computer
    return n**(1.0 + 0.05*n_cond) * (0.9 + 0.1*math.sqrt(k)) * 1e-5

def _map_maybe_parallel(func: Callable[[T], float], params: List[T],
    max_threads: Optional[int], time_estimate: float,
    callback: Callable[[int], None]) -> Iterable[float]:
    # If there is benefit in doing so, and the user has not overridden the
    # heuristic, execute the estimation in multiple parallel threads.
    # Multithreading is fine, because the estimator code releases the
    # Global Interpreter Lock for most of the time. Because the threads are
    # CPU bound, we should use at most as many threads as there are cores.

    # If the total execution time is very small, do not bother with threading
    if len(params) * time_estimate < 0.2:
        num_threads = 1
    else:
        num_threads = cpu_count() or 1

    if max_threads is not None:
        num_threads = min(num_threads, max_threads)

    if num_threads > 1:
        # Submit the individual tasks to thread pool
        result = [ np.nan ] * len(params)
        with concurrent.futures.ThreadPoolExecutor(num_threads, "ennemi-work") as executor:
            tasks = []

            # Do some work that would be done by executor.map() so that we can
            # implement callbacks. The result list must be in the same order
            # as the params list.
            # Type comments because type info for Future comes from typeshed;
            # the class itself is not subscriptable
            def get_callback(i): # type: (int) -> Callable[[concurrent.futures.Future[float]], None]
                def done(future): # type: (concurrent.futures.Future[float]) -> None
                    result[i] = future.result()
                    callback(i)
                return done
            
            for (i, p) in enumerate(params):
                task = executor.submit(func, p)
                task.add_done_callback(get_callback(i))
                tasks.append(task)

            concurrent.futures.wait(tasks)
        return result
    else:
        # Run the tasks sequentially
        # To support callbacks, we reimplement map() here too
        result = []
        for (i, p) in enumerate(params):
            result.append(func(p))
            callback(i)
        return result


def _lagged_mi(param_tuple: Tuple[np.ndarray, np.ndarray, int, int, int, int,
        Optional[np.ndarray], Optional[np.ndarray], int, bool, bool, bool]) -> float:
    # Unpack the param tuple used for possible cross-thread transfer
    x, y, lag, max_lag, min_lag, k, mask, cond, cond_lag, discrete_y, preprocess, drop_nan = param_tuple

    # Handle negative lags correctly
    min_lag = min(min_lag, 0)
    max_lag = max(max_lag, 0)

    # The x observations start from max_lag - lag
    xs = x[max_lag-lag : len(x)-lag+min_lag]
    # The y observations always start from max_lag
    ys = y[max_lag : len(y)+min_lag]
    # The cond observations have their own lag term
    if cond is not None:
        zs = cond[max_lag-cond_lag : len(cond)-cond_lag+min_lag]
    else: zs = None

    # Apply masks, validate and preprocess the data
    xs, ys, zs = _apply_masks(xs, ys, zs, mask, min_lag, max_lag, drop_nan)
    _validate_masked_data(xs, ys, zs, k, discrete_y)
    if preprocess:
        xs, ys, zs = _rescale_data(xs, ys, zs, discrete_y)
    
    # Apply the relevant estimation method
    if cond is None:
        if discrete_y:
            return _estimate_semidiscrete_mi(xs, ys, k)
        else:
            return _estimate_single_mi(xs, ys, k)
    else:
        if discrete_y:
            return _estimate_conditional_semidiscrete_mi(xs, ys, zs, k)
        else:
            return _estimate_conditional_mi(xs, ys, zs, k)

def _apply_masks(xs: np.ndarray, ys: np.ndarray, zs: Optional[np.ndarray],
        mask: np.ndarray, min_lag: int, max_lag: int, drop_nan: bool)\
        -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    # Apply the mask
    if mask is not None:
        mask_subset = mask[max_lag : len(mask)+min_lag]
        xs = xs[mask_subset]
        ys = ys[mask_subset]
        if zs is not None:
            zs = zs[mask_subset]

    # Drop NaNs
    if drop_nan:
        notna = ~(np.isnan(xs) | np.isnan(ys))
        if zs is not None:
            notna &= ~np.max(np.isnan(zs), axis=1)
            zs = zs[notna]
        xs = xs[notna]
        ys = ys[notna]

    return xs, ys, zs

def _validate_masked_data(xs: np.ndarray, ys: np.ndarray, zs: Optional[np.ndarray],
        k: int, discrete_y: bool) -> None:
    # Check that there are enough observations and no NaNs
    # Disable the check if y is discrete and non-numeric
    if (len(ys) <= k):
        raise ValueError("k must be smaller than number of observations (after lag and mask)")

    NAN_MSG = "input contains NaNs (after applying the mask), pass drop_nan=True to ignore"
    if np.isnan(xs).any():
        raise ValueError(NAN_MSG)
    if (not discrete_y or ys.dtype.kind in "iufc") and np.isnan(ys).any():
        raise ValueError(NAN_MSG)
    if zs is not None and np.isnan(zs).any():
        raise ValueError(NAN_MSG)

def _rescale_data(xs: np.ndarray, ys: np.ndarray, zs: Optional[np.ndarray], discrete_y: bool)\
        -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    # Digits of e for reproducibility
    rng = np.random.default_rng(2_718281828)

    # This warns if the standard deviation is zero
    xs = (xs - xs.mean()) / xs.std()
    xs += rng.normal(0.0, 1e-10, xs.shape)

    if not discrete_y:
        ys = (ys - ys.mean()) / ys.std()
        ys += rng.normal(0.0, 1e-10, ys.shape)
    
    if zs is not None:
        zs = (zs - zs.mean(axis=0)) / zs.std(axis=0)
        zs += rng.normal(0.0, 1e-10, zs.shape)

    return xs, ys, zs
