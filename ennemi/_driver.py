"""The one-line interface to this library.

Do not import this module directly, but rather import the main ennemi module.
"""

import concurrent.futures
from typing import List, Optional, Sequence, Tuple, TypeVar, Union
import itertools
import numpy as np # type: ignore
import sys
from ._entropy_estimators import _estimate_single_mi, _estimate_conditional_mi,\
    _estimate_single_entropy_1d, _estimate_single_entropy_nd

ArrayLike = Union[List[float], List[Tuple[float, ...]], np.ndarray]
GenArrayLike = TypeVar("GenArrayLike", List[float], List[Tuple[float, ...]], np.ndarray)

def normalize_mi(mi: GenArrayLike) -> GenArrayLike:
    """Normalize mutual information values to the unit interval.

    The return value matches the correlation coefficient between two Gaussian
    random variables with unit variance. This coefficient is preserved by most
    transformations, including scaling. The return value is positive regardless
    of the sign of the correlation.

    Negative values are kept as is. This is because mutual information is always
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
        import pandas # type: ignore
        if isinstance(mi, (pandas.DataFrame, pandas.Series)):
            return mi.applymap(_normalize)
    
    return np.vectorize(_normalize, otypes=[np.float])(mi)

def _normalize(mi: np.float) -> np.float:
    if mi <= 0.0:
        return mi
    else:
        return np.sqrt(1 - np.exp(-2 * mi))


def estimate_entropy(x: ArrayLike, *, k: int = 3, multidim: bool = False,
    mask: Optional[ArrayLike] = None, cond: Optional[ArrayLike] = None) -> np.ndarray:
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
    k : int
        The number of neighbors to consider. Default 3.
        Must be smaller than the number of observations left after masking.
    multidim : bool
        If False (the default), each column of `x` is considered a separate variable.
        If True, the (n x m) array is considered a single m-dimensional variable.
    mask : array_like or None
        If specified, an array of booleans that gives the input elements to use for
        estimation. Use this to exclude some observations from consideration.
        The length of this array must match the length of `x`.
        Currently, the mask must be one-dimensional.
    cond : array_like or None
        Optional 1D or 2D array of observations used for conditioning.
        Must have as many observations as `x`.
        All variables in a 2D array are used together.
        The calculation uses the chain rule H(X|Y) = H(X,Y) - H(Y) without
        any correction for potential estimation bias.
    """

    x_arr = np.asarray(x)

    # Apply the mask
    # TODO: This must be moved if each variable gets a separate mask
    if mask is not None:
        mask = np.asarray(mask)
        _validate_mask(mask, x_arr.shape[0])
        x_arr = x_arr[mask]

    # Validate the parameters
    if not 1 <= x_arr.ndim <= 2:
        raise ValueError("x must be one- or two-dimensional")
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if k >= x_arr.shape[0]:
        raise ValueError("k must be smaller than number of observations (after lag and mask)")
    if np.any(np.isnan(x_arr)):
        raise ValueError("input contains NaNs (after applying the mask)")

    if cond is None:
        result = _estimate_entropy(x_arr, k, multidim)
    else:
        cond_arr = np.asarray(cond)
        _validate_cond(cond_arr, x_arr.shape[0])
        result = _estimate_conditional_entropy(x_arr, cond_arr, k, multidim)

    # If the original x array was a pandas data type, return a DataFrame
    # As an exception, if multidim=True, we still return a NumPy scalar
    if not multidim and "pandas" in sys.modules:
        import pandas
        if isinstance(x, pandas.DataFrame):
            return pandas.DataFrame(np.atleast_2d(result), columns=x.columns)
        elif isinstance(x, pandas.Series):
            return pandas.DataFrame(np.atleast_2d(result), columns=[x.name])
    return result


def _estimate_entropy(x: np.ndarray, k: int, multidim: bool) -> np.ndarray:
    """Strongly typed estimate_entropy()."""

    if multidim and x.ndim > 1:
        return np.asarray(_estimate_single_entropy_nd(x, k))
    elif x.ndim == 1:
        return np.asarray(_estimate_single_entropy_1d(x, k))
    else:
        nvar = x.shape[1]
        return np.asarray([_estimate_single_entropy_1d(x[:,i], k) for i in range(nvar)])


def _estimate_conditional_entropy(x: np.ndarray, cond: np.ndarray, k: int, multidim: bool) -> np.ndarray:
    """Conditional entropy by the chain rule: H(X|Y) = H(X,Y) - H(Y)."""

    # Estimate the entropy of cond by the method above
    marginal = _estimate_entropy(cond, k, multidim=True)

    # The joint entropy depends on multidim and number of dimensions:
    # In the latter case, the joint entropy is calculated for each x variable
    if multidim or x.ndim == 1:
        joint = _estimate_single_entropy_nd(np.column_stack((x, cond)), k)
        return np.asarray(joint - marginal)
    else:
        nvar = x.shape[1]
        joint = np.asarray(
            [_estimate_single_entropy_nd(np.column_stack((x[:,i], cond)), k) for i in range(nvar)])
        return joint - marginal


def estimate_mi(y: ArrayLike, x: ArrayLike,
                lag: Union[Sequence[int], np.ndarray, int] = 0,
                *, k: int = 3,
                cond: Optional[ArrayLike] = None,
                cond_lag: Union[Sequence[int], np.ndarray, int] = 0,
                mask: Optional[ArrayLike] = None,
                normalize: bool = False,
                parallel: Optional[str] = None) -> np.ndarray:
    """Estimate the mutual information between y and each x variable.
 
    Returns the estimated mutual information (in nats) for continuous
    variables. The result is a 2D `ndarray` where the first index represents `x`
    rows and the second index represents the `lag` values. If `x` is a pandas
    `DataFrame` or `Series`, the result is a `DataFrame`.

    The time lag is interpreted as `y(t) ~ x(t - lag) | z(t - cond_lag)`.
    The time lags are applied to the `x` and `cond` arrays such that the `y`
    array stays the same every time.
    This means that `y` is cropped to `y[max_lag:N+min(min_lag, 0)]`.

    If the `cond` parameter is set, conditional mutual information is estimated.
    The `cond_lag` parameter specifies the lag for the `cond` array, separately
    from the `x` lag.

    If the `mask` parameter is set, only those `y` observations with the
    matching mask element set to `True` are used for estimation.
    
    If the data set contains many identical observations,
    this method may return incorrect results or `-inf`.
    In that case, add low-amplitude noise to the data and try again.

    The calculation is based on Kraskov et al. (2004): Estimating mutual
    information. Physical Review E 69. doi:10.1103/PhysRevE.69.066138

    Positional or keyword parameters:
    ---
    y : array_like
        A 1D array of observations.
    x : array_like
        A 1D or 2D array where the columns are one or more variables and the
        rows are observations. The number of rows must be the same as in y.
    lag : int or array_like
        A time lag or 1D array of time lags to apply to x. Default 0.
        The values may be any integers with magnitude
        less than the number of observations.

    Optional keyword parameters:
    ---
    k : int
        The number of neighbors to consider. Default 3.
        Must be smaller than the number of observations left after masking and cropping.
    cond : array_like or None
        Optional 1D or 2D array of observations used for conditioning.
        Must have as many observations as y.
        All variables in a 2D array are used together.
    cond_lag : int
        Lag applied to the cond array.
        Must be broadcastable to the size of `lag`. Default 0.
    mask : array_like or None
        If specified, an array of booleans that gives the y elements to use for
        estimation. Use this to exclude some observations from consideration
        while preserving the time series structure of the data. Elements of
        `x` and `cond` are masked with the lags applied. The length of this
        array must match the length `y`.
    normalize : bool
        If True, the results will be normalized to correlation coefficient scale.
        Same as calling `normalize_mi` on the results.
    parallel : str or None
        Whether to run the estimation in multiple processes. If None (the default),
        a heuristic will be used for the decision. If "always", the estimation
        will be run on as many concurrent processes as there are processors.
        If "disable", the combinations are estimated sequentially in the current process.
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
    result = _estimate_mi(y_arr, x_arr, lag_arr, k, cond_arr, cond_lag_arr, mask_arr, parallel)

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
        mask: Optional[np.ndarray], parallel: Optional[str]) -> np.ndarray:
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
    # The params map contains tuples for simpler passing into subprocess
    indices = list(itertools.product(range(len(lag)), range(nvar)))
    if x.ndim == 1:
        params = map(lambda i: (x, y, lag[i[0]], max_lag, min_lag, k, mask, cond, cond_lag[i[0]]), indices)
    else:
        params = map(lambda i: (x[:,i[1]], y, lag[i[0]], max_lag, min_lag, k, mask, cond, cond_lag[i[0]]), indices)

    # If there is benefit in doing so, and the user has not overridden the
    # heuristic, execute the estimation in multiple parallel processes
    if _should_be_parallel(parallel, len(indices), len(y)):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            conc_result = executor.map(_lagged_mi, params)
    else:
        conc_result = map(_lagged_mi, params)
    
    # Collect the results to a 2D array
    result = np.empty((len(lag), nvar))
    for index, res in zip(indices, conc_result):
        result[index] = res
    return result


def _check_parameters(x: np.ndarray, y: Optional[np.ndarray], k: int,
        cond: Optional[np.ndarray], mask: Optional[np.ndarray]) -> None:
    """Does most of parameter checking, but some is still left to _lagged_mi."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

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


def pairwise_mi(data: ArrayLike, *, k: int = 3, cond: Optional[ArrayLike] = None,
    mask: Optional[ArrayLike] = None, parallel: Optional[str] = None,
    normalize: bool = False) -> np.ndarray:
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
    k : int
        The number of neighbors to consider. Default 3.
        Must be smaller than the number of observations left after masking.
    cond : array_like or None
        Optional 1D or 2D array of observations used for conditioning.
        Must have as many observations as the data.
        All variables in a 2D array are used together.
    mask : array_like or None
        If specified, an array of booleans that gives the data elements to use for
        estimation. Use this to exclude some observations from consideration
        while preserving the time series structure of the data. The length of
        this array must match the length of `data`.
    normalize: bool
        If True, the MI values will be normalized to correlation coefficient scale.
    parallel : str or None
        Whether to run the estimation in multiple processes. If None (the default),
        a heuristic will be used for the decision. If "always", the estimation
        will be run on as many concurrent processes as there are processors.
        If "disable", the combinations are estimated sequentially in the current process.
    """
    data_arr = np.asarray(data)
    if cond is not None: cond_arr = np.asarray(cond)
    else: cond_arr = None
    if mask is not None: mask_arr = np.asarray(mask)
    else: mask_arr = None

    # If there is just one variable, return the trivial result
    if data_arr.ndim == 1 or data_arr.shape[1] == 1:
        return np.full((1,1), np.nan)
    
    result = _pairwise_mi(data_arr, k, cond_arr, mask_arr, parallel)

    # Normalize if asked for
    if normalize:
        result = normalize_mi(result)

    # If data was a pandas DataFrame, return a DataFrame with matching names
    if "pandas" in sys.modules:
        import pandas
        if isinstance(data, pandas.DataFrame):
            return pandas.DataFrame(result, index=data.columns, columns=data.columns)
    return result


def _pairwise_mi(data: np.ndarray, k: int, cond: Optional[np.ndarray],
    mask: Optional[np.ndarray], parallel: Optional[str]) -> np.ndarray:
    """Strongly typed pairwise MI. The data array is at least 2D."""

    _check_parameters(data, None, k, cond, mask)
    nvar = data.shape[1]

    # Create a list of variable pairs
    # By symmetry, it suffices to consider a triangular matrix
    indices = []
    params = []
    for i in range(nvar):
        for j in range(i+1, nvar):
            indices.append((i, j))
            params.append((data[:,i], data[:,j], 0, 0, 0, k, mask, cond, 0))

    # Run the MI estimation for each pair, possibly in parallel
    if _should_be_parallel(parallel, len(indices), data.shape[0]):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            conc_result = executor.map(_lagged_mi, params)
    else:
        conc_result = map(_lagged_mi, params)

    # Collect the results, creating a symmetric matrix now
    result = np.full((nvar, nvar), np.nan)
    for (i,j), res in zip(indices, conc_result):
        result[i,j] = res
        result[j,i] = res
    
    return result


def _should_be_parallel(parallel: Optional[str], num_cases: int, num_obs: int) -> bool:
    # Check whether the user has forced a certain parallel mode
    if parallel == "always":
        return True
    elif parallel == "disable":
        return False
    elif parallel is not None:
        raise ValueError("unrecognized value for parallel argument")
    else:
        # As the user has not overridden the choice, use a heuristic
        # TODO: In a many variables/lags, small N case, it may make sense to
        #       use multiple processes, but batch the tasks
        return num_cases > 1 and num_obs > 200


def _lagged_mi(param_tuple: Tuple[np.ndarray, np.ndarray, int, int, int, int,
        Optional[np.ndarray], Optional[np.ndarray], int]) -> float:
    # Unpack the param tuple used for possible cross-process transfer
    x, y, lag, max_lag, min_lag, k, mask, cond, cond_lag = param_tuple

    # Handle negative lags correctly
    min_lag = min(min_lag, 0)
    max_lag = max(max_lag, 0)

    # The x observations start from max_lag - lag
    xs = x[max_lag-lag : len(x)-lag+min_lag]
    # The y observations always start from max_lag
    ys = y[max_lag : len(y)+min_lag]

    # Mask the observations if necessary
    if mask is not None:
        mask_subset = mask[max_lag : len(y)+min_lag]
        xs = xs[mask_subset]
        ys = ys[mask_subset]

    # Check that there are enough observations and no NaNs
    if (len(ys) <= k):
        raise ValueError("k must be smaller than number of observations (after lag and mask)")
    if np.isnan(xs).any() or np.isnan(ys).any():
        raise ValueError("input contains NaNs (after applying the mask)")
    
    if cond is None:
        return _estimate_single_mi(xs, ys, k)
    else:
        # The cond observations have their own lag term
        zs = cond[max_lag-cond_lag : len(cond)-cond_lag+min_lag]
        if mask is not None:
            zs = zs[mask_subset]

        if np.isnan(zs).any():
            raise ValueError("input contains NaNs (after applying the mask)")

        return _estimate_conditional_mi(xs, ys, zs, k)
