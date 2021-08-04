# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

"""The one-line interface to this library.

Do not import this module directly, but rather import the main ennemi module.
"""

from __future__ import annotations
import concurrent.futures
from typing import Callable, Iterable, Optional, Sequence, Tuple, Type, TypeVar, Union
import itertools
import math
import numpy as np
from os import cpu_count
import sys
from ._entropy_estimators import _estimate_single_mi, _estimate_conditional_mi,\
    _estimate_discrete_mi, _estimate_conditional_discrete_mi,\
    _estimate_semidiscrete_mi, _estimate_conditional_semidiscrete_mi,\
    _estimate_single_entropy, _estimate_discrete_entropy

try:
    import numpy.typing as npt
    FloatArray = npt.NDArray[np.float64]
    ArrayLike = npt.ArrayLike
except:
    # Just stop supporting annotations altogether on NumPy 1.20 and below
    FloatArray = np.ndarray # type: ignore
    ArrayLike = np.ndarray # type: ignore
GenArrayLike = TypeVar("GenArrayLike", Sequence[float], Sequence[Sequence[float]], FloatArray)
T = TypeVar("T")

def normalize_mi(mi: Union[float, GenArrayLike]) -> GenArrayLike:
    """Normalize mutual information values to the unit interval.

    Equivalent to passing `normalize=True` to the estimation methods.

    The return value matches the correlation coefficient between two Gaussian
    random variables with unit variance. This coefficient is preserved by all
    monotonic transformations, including scaling. The value is positive regardless
    of the sign of the correlation.

    Negative values are kept as-is. This is because mutual information is always
    non-negative, but `estimate_mi` may produce negative values.

    The normalization is not applicable to discrete variables:
    it is not possible to get coefficient 1.0 even when the variables are completely
    determined by each other. The formula assumes that the both variables have
    an infinite amount of entropy.

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
    
    return np.vectorize(_normalize, otypes=[float])(mi)

def _normalize(mi: float) -> float:
    if mi <= 0.0:
        return mi
    else:
        return np.sqrt(1 - np.exp(-2 * mi))


def estimate_entropy(x: ArrayLike,
    *, k: int = 3,
    multidim: bool = False,
    discrete: bool = False,
    mask: Optional[ArrayLike] = None,
    cond: Optional[ArrayLike] = None,
    drop_nan: bool = False) -> FloatArray:
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
    multidim : bool, default False
        If False (the default), each column of `x` is considered a separate variable.
        If True, the (n x m) array is considered a single m-dimensional variable.
    discrete : bool, default False
        If True, the variable and the optional conditioning variable are interpreted
        as discrete variables. The result will be calculated using the mathematical
        definition of entropy.
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
        If True, all NaN (not a number) values in `x` and `cond` are masked out.
        Not applied to discrete data.
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
        result = _estimate_entropy(x_arr, k, multidim, mask, discrete, drop_nan)
    else:
        cond_arr = np.asarray(cond)
        _validate_cond(cond_arr, x_arr.shape[0])
        result = _estimate_conditional_entropy(x_arr, cond_arr, k, multidim, mask, discrete, drop_nan)

    # If the original x array was a pandas data type, return a DataFrame
    # As an exception, if multidim=True, we still return a NumPy scalar
    if not multidim and "pandas" in sys.modules:
        import pandas
        if isinstance(x, pandas.DataFrame):
            return pandas.DataFrame(np.atleast_2d(result), columns=x.columns)
        elif isinstance(x, pandas.Series):
            return pandas.DataFrame(np.atleast_2d(result), columns=[x.name])
    return result


def _estimate_entropy(x: FloatArray, k: int, multidim: bool,
        mask: Optional[ArrayLike], discrete: bool, drop_nan: bool) -> FloatArray:
    """Strongly typed estimate_entropy()."""

    if multidim or x.ndim == 1:
        x = _mask_and_validate_entropy(x, mask, drop_nan, discrete, k)
        return np.asarray(_call_entropy_func(x, k, discrete))
    else:
        nvar = x.shape[1]
        result = np.empty(nvar)
        for i in range(nvar):
            xs = _mask_and_validate_entropy(x[:,i], mask, drop_nan, discrete, k)
            result[i] = _call_entropy_func(xs, k, discrete)
        return result

def _mask_and_validate_entropy(x: FloatArray, mask: Optional[ArrayLike],
        drop_nan: bool, discrete: bool, k: int) -> FloatArray:
    # Apply the mask and drop NaNs
    # TODO: Support 2D masks (https://github.com/polsys/ennemi/issues/37)
    if mask is not None:
        x = x[mask]

    if drop_nan and not discrete:
        if x.ndim > 1:
            x = x[~np.max(np.isnan(x), axis=1)]
        else:
            x = x[~np.isnan(x)]

    # Validate the x array
    if k >= x.shape[0]:
        raise ValueError("k must be smaller than number of observations (after lag and mask)")
    if not discrete and np.any(np.isnan(x)):
        raise ValueError("input contains NaNs (after applying the mask), pass drop_nan=True to ignore")

    return x

def _estimate_conditional_entropy(x: FloatArray, cond: FloatArray, k: int, multidim: bool,
        mask: Optional[ArrayLike], discrete: bool, drop_nan: bool) -> FloatArray:
    """Conditional entropy by the chain rule: H(X|Y) = H(X,Y) - H(Y)."""

    # Estimate the entropy of cond by the method above (multidim=True)
    marginal = _estimate_entropy(cond, k, True, mask, discrete, drop_nan)

    # The joint entropy depends on multidim and number of dimensions:
    # In the latter case, the joint entropy is calculated for each x variable
    if multidim or x.ndim == 1:
        xs = _mask_and_validate_entropy(np.column_stack((x, cond)), mask, drop_nan, discrete, k)
        return np.asarray(_call_entropy_func(xs, k, discrete) - marginal)
    else:
        nvar = x.shape[1]
        joint = np.empty(nvar) # type: npt.NDArray[np.float64]
        for i in range(nvar):
            xs = _mask_and_validate_entropy(np.column_stack((x[:,i], cond)), mask, drop_nan, discrete, k)
            joint[i] = _call_entropy_func(xs, k, discrete)
        return joint - marginal

def _call_entropy_func(xs: FloatArray, k: int, discrete: bool) -> float:
    if discrete:
        return _estimate_discrete_entropy(xs, k)
    else:
        return _estimate_single_entropy(xs, k)


def estimate_mi(y: ArrayLike, x: ArrayLike,
                lag: Union[Sequence[int], ArrayLike, int] = 0,
                *, k: int = 3,
                cond: Optional[ArrayLike] = None,
                cond_lag: Union[Sequence[int], Sequence[Sequence[int]], ArrayLike, int] = 0,
                mask: Optional[ArrayLike] = None,
                discrete_y: bool = False,
                discrete_x: bool = False,
                preprocess: bool = True,
                drop_nan: bool = False,
                normalize: bool = False,
                max_threads: Optional[int] = None,
                callback: Optional[Callable[[int, int], None]] = None) -> FloatArray:
    """Estimate the mutual information between y and each x variable.

    - Unconditional MI: the default.
    - Conditional MI: pass a `cond` array.
    - Discrete-continuous MI: set `discrete_y` to True.
 
    Returns the estimated mutual information (in nats) for continuous
    variables. The result is a 2D `ndarray` where the first index represents `lag` values
    and the second index represents `x` columns. If `x` is a pandas
    `DataFrame` or `Series`, the result is a `DataFrame`.

    The time lag is interpreted as `y(t) ~ x(t - lag) | z(t - cond_lag)`.
    The time lags are applied to the `x` and `cond` arrays such that the `y`
    array stays the same every time.
    This means that `y` is cropped to `y[max(max_lag,0) : N+min(min_lag,0)]`.
    The `cond_lag` parameter specifies the lag for the `cond` array separately
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
        A time lag or 1D array of time lags to apply. A positive lag means that
        `y` depends on earlier `x` values and vice versa.

    Optional keyword parameters:
    ---
    k : int, default 3
        The number of neighbors to consider.
        Ignored if both variables are discrete.
    cond : array_like or None
        Optional 1D or 2D array of observations used for conditioning.
        Must have as many observations as y.
        All variables in a 2D array are used together.
        If both `discrete_x` and `discrete_y` are set, this is interpreted as a
        discrete variable. (In future versions, this might be set explicitly.)
    cond_lag : int or array_like, default 0
        Lag applied to the cond array. Must be broadcastable to the size of `lag`.
        Can be two-dimensional to lag each conditioning variable separately.
    mask : array_like or None
        If specified, an array of booleans that gives the `y` elements to use for
        estimation. Use this to exclude some observations from consideration
        while preserving the time series structure of the data. Elements of
        `x` and `cond` are masked with the lags applied.
    discrete_x : bool, default False
    discrete_y : bool, default False
        If True, the respective variable is interpreted as a discrete variable.
        Values of the variable may be non-numeric.
    preprocess : bool, default True
        By default, continuous variables are scaled to unit variance and
        added with low-amplitude noise. The noise uses a fixed random seed.
    drop_nan : bool, default False
        If True, all NaN (not a number) values are masked out.
    normalize : bool, default False
        If True, the results will be normalized to correlation coefficient scale.
        Same as calling `normalize_mi` on the results.
        The results are sensible only if both variables are continuous.
    max_threads : int or None
        The maximum number of threads to use for estimation.
        By default, the number of CPU cores is used.
    callback : method or None
        A method to call when each estimation task is completed. The method
        must take two integer parameters: `x` variable index and lag value.
        This method should be very short. Because Python code is not executed
        concurrently, the callback may slow down other estimation tasks.
    """
    
    # Convert parameters to consistent types
    # Keep the original x parameter around for the Pandas data frame check
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    if cond is not None:
        # Make cond 2D
        cond_arr = np.column_stack((np.asarray(cond),))
        ncond = cond_arr.shape[1]
    else:
        cond_arr = None
        ncond = 1
    mask_arr = None # type: Optional[npt.NDArray[np.float64]]
    if mask is not None: mask_arr = np.asarray(mask)

    # Broadcast cond_lag to be (#lags, #cond vars) in shape
    lag_arr = np.atleast_1d(lag)
    cond_lag_arr = np.broadcast_to(np.column_stack((cond_lag,)), (lag_arr.shape[0], ncond))

    # Check the parameters and run the estimation
    result = _estimate_mi(y_arr, x_arr, lag_arr, k, cond_arr,
        cond_lag_arr, mask_arr, discrete_x, discrete_y, preprocess, drop_nan, max_threads, callback)

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

def _estimate_mi(y: FloatArray, x: FloatArray, lag: FloatArray, k: int,
        cond: Optional[FloatArray], cond_lag: FloatArray,
        mask: Optional[FloatArray], discrete_x: bool, discrete_y: bool,
        preprocess: bool, drop_nan: bool,
        max_threads: Optional[int],
        callback: Optional[Callable[[int, int], None]]) -> FloatArray:
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
            mask, cond, cond_lag[i[0]], discrete_x, discrete_y, preprocess, drop_nan), indices))
    else:
        params = list(map(lambda i: (x[:,i[1]], y, lag[i[0]], max_lag, min_lag, k,
            mask, cond, cond_lag[i[0]], discrete_x, discrete_y, preprocess, drop_nan), indices))

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


def _check_parameters(x: FloatArray, y: Optional[FloatArray], k: int,
        cond: Optional[FloatArray], mask: Optional[FloatArray]) -> None:
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

def _validate_mask(mask: FloatArray, input_len: int) -> None:
    if len(mask.shape) > 1:
        raise ValueError("mask must be one-dimensional")
    if len(mask) != input_len:
        raise ValueError("mask length does not match input length")
    if mask.dtype != bool:
        raise TypeError("mask must contain only booleans")

def _validate_cond(cond: FloatArray, input_len: int) -> None:
    if not 1 <= cond.ndim <= 2:
        raise ValueError("cond must be one- or two-dimensional")
    if input_len != len(cond):
        raise ValueError("x and cond must have same length")


def pairwise_mi(data: ArrayLike,
    *, k: int = 3,
    cond: Optional[ArrayLike] = None,
    mask: Optional[ArrayLike] = None,
    discrete: ArrayLike = False,
    preprocess: bool = True,
    drop_nan: bool = False,
    normalize: bool = False,
    max_threads: Optional[int] = None,
    callback: Optional[Callable[[int, int], None]] = None) -> FloatArray:
    """Estimate the pairwise MI between each variable.

    Returns a matrix where the (i,j)'th element is the mutual information
    between the i'th and j'th columns in the data. The values are in nats or
    in the normalized scale depending on the `normalize` parameter. The diagonal
    contains NaNs (for better visualization, as the auto-MI should be infinite).

    Positional or keyword parameters:
    ---
    data : array_like
        A 2D array where the columns represent variables.

    Optional keyword parameters:
    ---
    k : int, default 3
        The number of neighbors to consider.
        Ignored if both variables are discrete.
    cond : array_like or None
        Optional 1D or 2D array of observations used for conditioning.
        Must have as many observations as the data.
        All variables in a 2D array are used together.
    mask : array_like or None
        If specified, an array of booleans that gives the data elements to use for
        estimation. Use this to exclude some observations from consideration.
    discrete : bool or array_like, default False
        Specifies the columns that contain discrete data. Conditioning can be
        used only if the data is all-continuous or all-discrete (in which case
        the condition is interpreted as discrete).
    preprocess : bool, default True
        By default, continuous variables are scaled to unit variance and
        added with low-amplitude noise. The noise uses a fixed random seed.
    drop_nan : bool, default False
        If True, all NaN (not a number) values in `x` and `cond` are masked out.
    normalize : bool, default False
        If True, the MI values will be normalized to correlation coefficient scale.
        Same as calling `normalize_mi` on the results.
        The results are sensible only if both variables are continuous.
    max_threads : int or None
        The maximum number of threads to use for estimation.
        By default, the number of CPU cores is used.
    callback : method or None
        A method to call when each estimation task is completed. The method
        must take two integer parameters, representing the variable indices.
        This method should be very short. Because Python code is not executed
        concurrently, the callback may slow down other estimation tasks.
    """

    # Convert arrays to consistent type; _lagged_mi assumes cond to be 2D
    data_arr = np.asarray(data)
    cond_arr = None # type: Optional[npt.NDArray[np.float64]]
    mask_arr = None # type: Optional[npt.NDArray[np.float64]]
    if cond is not None: cond_arr = np.column_stack((np.asarray(cond),))
    if mask is not None: mask_arr = np.asarray(mask)

    # If there is just one variable, return the trivial result
    if data_arr.ndim == 1 or data_arr.shape[1] == 1:
        return np.full((1,1), np.nan)

    discrete_arr = np.broadcast_to(discrete, data_arr.shape[1])
    
    result = _pairwise_mi(data_arr, k, cond_arr, preprocess, drop_nan, mask_arr,
        discrete_arr, max_threads, callback)

    # Normalize if asked for
    if normalize:
        result = normalize_mi(result)

    # If data was a pandas DataFrame, return a DataFrame with matching names
    if "pandas" in sys.modules:
        import pandas
        if isinstance(data, pandas.DataFrame):
            return pandas.DataFrame(result, index=data.columns, columns=data.columns)
    return result


def _pairwise_mi(data: FloatArray, k: int, cond: Optional[FloatArray], preprocess: bool,
    drop_nan: bool, mask: Optional[FloatArray], discrete: FloatArray, max_threads: Optional[int],
    callback: Optional[Callable[[int, int], None]]) -> FloatArray:
    """Strongly typed pairwise MI. The data array is at least 2D."""

    _check_parameters(data, None, k, cond, mask)
    if (not (np.all(discrete) or np.all(~discrete))) and (cond is not None):
        raise ValueError("Conditioning is not supported with mixed discrete and continuous data. " +
            "This is a limitation that can be lifted in the future (see " +
            "https://github.com/polsys/ennemi/issues/87).")

    nobs, nvar = data.shape

    # Fake a cond_lag of correct shape for _lagged_mi
    # TODO: Actually support cond lag (https://github.com/polsys/ennemi/issues/61)
    if cond is not None:
        cond_lag = np.full(cond.shape[1], 0)
    else: cond_lag = np.asarray(0)

    # Create a list of variable pairs
    # By symmetry, it suffices to consider a triangular matrix
    indices = []
    params = []
    for i in range(nvar):
        for j in range(i+1, nvar):
            indices.append((i, j))
            params.append((data[:,i], data[:,j], 0, 0, 0, k, mask, cond, cond_lag,
                discrete[i], discrete[j], preprocess, drop_nan))

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


def _get_mi_time_estimate(n: int, cond: Optional[FloatArray], k: int) -> float:
    if cond is None:
        n_cond = 0
    else:
        # cond is guaranteed to be two-dimensional
        n_cond = cond.shape[1]

    # These are determined pretty experimentally on a laptop computer
    return n**(1.0 + 0.05*n_cond) * (0.9 + 0.1*math.sqrt(k)) * 1e-5

def _map_maybe_parallel(func: Callable[[T], float], params: Sequence[T],
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


def _lagged_mi(param_tuple: Tuple[FloatArray, FloatArray, int, int, int, int,
        Optional[FloatArray], Optional[FloatArray], FloatArray, bool, bool, bool, bool]) -> float:
    # Unpack the param tuple used for possible cross-thread transfer
    x, y, lag, max_lag, min_lag, k, mask, cond, cond_lag,\
        discrete_x, discrete_y, preprocess, drop_nan = param_tuple

    # Handle negative lags correctly
    min_lag = min(min_lag, 0)
    max_lag = max(max_lag, 0)

    # The x observations start from max_lag - lag
    xs = x[max_lag-lag : len(x)-lag+min_lag]
    # The y observations always start from max_lag
    ys = y[max_lag : len(y)+min_lag]
    # The cond observations have their own lag terms
    if cond is not None:
        zs = np.column_stack([cond[max_lag-cond_lag[i] : len(cond)-cond_lag[i]+min_lag, i]\
            for i in range(len(cond_lag))])
    else: zs = None

    # Apply masks, validate and preprocess the data
    xs, ys, zs = _apply_masks(xs, ys, zs, mask, min_lag, max_lag, drop_nan)
    _validate_masked_data(xs, ys, zs, k, discrete_x, discrete_y)
    if preprocess:
        xs, ys, zs = _rescale_data(xs, ys, zs, discrete_x, discrete_y)
    
    # Apply the relevant estimation method
    if cond is None:
        if discrete_x and discrete_y:
            return _estimate_discrete_mi(xs, ys)
        elif discrete_x:
            return _estimate_semidiscrete_mi(ys, xs, k)
        if discrete_y:
            return _estimate_semidiscrete_mi(xs, ys, k)
        else:
            return _estimate_single_mi(xs, ys, k)
    else:
        if discrete_x and discrete_y:
            return _estimate_conditional_discrete_mi(xs, ys, zs)
        elif discrete_x:
            return _estimate_conditional_semidiscrete_mi(ys, xs, zs, k)
        if discrete_y:
            return _estimate_conditional_semidiscrete_mi(xs, ys, zs, k)
        else:
            return _estimate_conditional_mi(xs, ys, zs, k)

def _apply_masks(xs: FloatArray, ys: FloatArray, zs: Optional[FloatArray],
        mask: Optional[FloatArray], min_lag: int, max_lag: int, drop_nan: bool)\
        -> Tuple[FloatArray, FloatArray, Optional[FloatArray]]:
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

def _validate_masked_data(xs: FloatArray, ys: FloatArray, zs: Optional[FloatArray],
        k: int, discrete_x: bool, discrete_y: bool) -> None:
    # Check that there are enough observations and no NaNs
    # Disable the check if y is discrete and non-numeric
    if (len(ys) <= k):
        raise ValueError("k must be smaller than number of observations (after lag and mask)")

    NAN_MSG = "input contains NaNs (after applying the mask), pass drop_nan=True to ignore"
    if (not discrete_x or xs.dtype.kind in "iufc") and np.isnan(xs).any():
        raise ValueError(NAN_MSG)
    if (not discrete_y or ys.dtype.kind in "iufc") and np.isnan(ys).any():
        raise ValueError(NAN_MSG)
    if not (discrete_x and discrete_y) and zs is not None and np.isnan(zs).any():
        raise ValueError(NAN_MSG)

def _rescale_data(xs: FloatArray, ys: FloatArray, zs: Optional[FloatArray],
        discrete_x: bool, discrete_y: bool) -> Tuple[FloatArray, FloatArray, Optional[FloatArray]]:
    # Digits of e for reproducibility
    rng = np.random.default_rng(2_718281828)

    if not discrete_x:
        # This warns if the standard deviation is zero
        xs = (xs - xs.mean()) / xs.std()
        xs += rng.normal(0.0, 1e-10, xs.shape)

    if not discrete_y:
        ys = (ys - ys.mean()) / ys.std()
        ys += rng.normal(0.0, 1e-10, ys.shape)
    
    # If both X and Y are discrete, we assume the condition to be discrete too
    if zs is not None and not (discrete_x and discrete_y):
        zs = (zs - zs.mean(axis=0)) / zs.std(axis=0)
        zs += rng.normal(0.0, 1e-10, zs.shape) # type: ignore # mypy does not realize this is ndarray

    return xs, ys, zs
