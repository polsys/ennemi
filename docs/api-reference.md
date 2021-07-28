---
title: API reference
---

This page is an extended version of the documentation strings included
for each method.
All methods are available in the `ennemi` namespace.



# Notes
The `array_like` data type is interpreted as in NumPy documentation:
it is either a `numpy.ndarray` or anything with roughly the same shape.
Python lists and tuples are array-like.
Two-dimensional arrays are created by nesting sequences:
`[(1, 2), (3, 4), (5, 6)]` is equivalent to an array with 3 rows and 2 columns.
Numbers are interpreted as zero-dimensional arrays.
Arrays can be automatically extended to higher dimensions if the shapes match.


The calculation algorithms are described in
- Kraskov et al. (2004): Estimating mutual information. Physical Review E 69.
  [doi:10.1103/PhysRevE.69.066138](https://dx.doi.org/10.1103/PhysRevE.69.066138).
- Frenzel & Pompe (2007): Partial Mutual Information for Coupling Analysis of
  Multivariate Time Series. Physical Review Letters 99.
  [doi:10.1103/PhysRevLett.99.204101](https://dx.doi.org/10.1103/PhysRevLett.99.204101).
- Ross (2014): Mutual Information between Discrete and Continuous Data Sets.
  PLoS ONE 9.
  [doi:10.1371/journal.pone.0087357](https://dx.doi.org/10.1371/journal.pone.0087357).

The algorithms assume that
- the data is continuous,
- the variables have roughly symmetric distributions,
- the observations are statistically independent and identically distributed.

Note especially the last assumption.
Highly autocorrelated time series data will produce unrealistically high MI values.



<hr style="margin: 4rem 0;">

# `estimate_entropy`
Estimate the entropy of one or more continuous random variables.

Returns the estimated entropy in nats.
If `x` is two-dimensional, each marginal variable is estimated separately by default.
If the `multidim` parameter is set to True,
the array is interpreted as a single $m$-dimensional random variable.

If `x` is a pandas `DataFrame` or `Series`, the result is a `DataFrame`
with column names matching `x`.

### Parameters
- `x: array_like`.
  
  A 1D or 2D array of observations. The interpretation of columns
  depends on the `multidim` parameter.

### Optional keyword parameters
- `k: int`, default 3.
  
  The number of neighbors to consider.

- `multidim: bool`, default False.
  
  If False, each column of `x` is considered a separate variable.
  If True, the $(n \times m)$ array is considered a single $m$-dimensional variable.

- `mask: array_like` or None.
  
  If specified, an array of booleans that gives the input elements to use for
  estimation. Use this to exclude some observations from consideration.

  Currently, the mask must be one-dimensional.
  (Issue [#37](https://github.com/polsys/ennemi/issues/37))

- `cond: array_like` or None.
  
  Optional 1D or 2D array of observations used for conditioning.
  A $(n \times m)$ array is interpreted as a single $m$-dimensional variable.
  
  The calculation uses the chain rule $H(X|Y) = H(X,Y) - H(Y)$ without
  any correction for potential (small) estimation bias.

- `drop_nan: bool`, default False.

  If True, all NaN (not a number) values in `x` and `cond` are masked out.



<hr style="margin: 4rem 0;">

# `estimate_mi`
Estimate the mutual information between `y` and each `x` variable.

- Unconditional MI: the default.
- Conditional MI: pass a `cond` array.
- Discrete-continuous MI: set `discrete_y` to True.

Returns the estimated mutual information (in nats or in correlation scale).
The result is a 2D `ndarray` where the first index represents `lag` values
and the second index represents `x` columns.
If `x` is a pandas `DataFrame` or `Series`, the result is a `DataFrame`
with column names and lag values as the row indices.

The time lag $\Delta$ is interpreted as $y(t) \sim x(t - \Delta) | z(t - \Delta_{\mathrm{cond}})$.
The time lags are applied to the `x` and `cond` arrays such that the `y`
array stays the same every time.
This means that `y` is cropped to `y[max(max_lag,0) : N+min(min_lag,0)]`.
The `cond_lag` parameter specifies the lag for the `cond` array separately
from the `x` lag.

If the `mask` parameter is set, only those `y` observations with the
matching mask element set to `True` are used for estimation.

### Positional or keyword parameters
- `y: array_like`.
  
  A 1D array of observations. If `discrete_y` is True, the values may be
  of any type. Otherwise the values must be numeric.

- `x: array_like`.

  A 1D or 2D array where the columns are one or more variables and the
  rows are observations. The number of rows must be the same as in y.

- `lag: array_like`, default 0.
  
  A time lag or 1D array of time lags to apply.
  A positive lag means that `y` depends on earlier `x` values and vice versa.

### Optional keyword parameters
- `k: int`, default 3.
  
  The number of neighbors to consider.

- `cond: array_like`, default None.
  
  Optional 1D or 2D array of observations used for conditioning.
  Must have as many observations as `y`.
  A $(n \times m)$ array is interpreted as a single $m$-dimensional variable.

- `cond_lag: array_like`, default 0.

  Lag applied to the cond array. Must be broadcastable to the size of `lag`.
  Can be two-dimensional to lag each conditioning variable separately.

- `mask: array_like` or None.

  If specified, an array of booleans that gives the `y` elements to use for
  estimation. Use this to exclude some observations from consideration
  while preserving the time series structure of the data. Elements of
  `x` and `cond` are masked with the lags applied.

- `discrete_x, discrete_y: bool`, default False.

  If True, the respective variable is interpreted as a discrete variable.
  Values of the variable may be non-numeric.

- `preprocess: bool`, default True.

  By default, the variables are scaled to unit variance and
  added with low-amplitude noise. The noise uses a fixed random seed.
  This is not applied to discrete variables.

- `drop_nan: bool`, default False.

  If True, all NaN (not a number) values in `x`, `y` and `cond` are masked out.

- `normalize: bool`, default False.

  If True, the results will be normalized to correlation coefficient scale.
  Same as calling `normalize_mi` on the results.

- `max_threads: int` or None.
  
  The maximum number of threads to use for estimation.
  By default, the number of CPU cores is used.
  Because the calculation is CPU-bound, more threads should not be used.

- `callback: method` or None.
  
  A method to call when each estimation task is completed. The method
  must take two integer parameters: `x` variable index and lag value.

  This method should be very short. Because Python code is not executed
  concurrently, the callback may slow down other estimation tasks.

  See also: [Example progress reporter](snippets.md).



<hr style="margin: 4rem 0;">

# `normalize_mi`
Normalize mutual information values to the unit interval.
Equivalent to passing `normalize=True` to the estimation methods.

The normalization formula
$$
\rho = \sqrt{1 - \exp(-2\, \mathrm{MI}(X;Y))}
$$
is based on the MI of bivariate normal distribution.
The value matches the absolute Pearson correlation coefficient in a linear model.
However, the value is preserved by all monotonic transformations;
in a non-linear model, it matches the Pearson correlation of the linearized model.
The value is positive regardless of the sign of the correlation.

Negative values are kept as-is.
This is because mutual information is always
non-negative, but `estimate_mi` may produce negative values.
Large negative values may indicate that the data does not satisfy assumptions.

### Parameters
- `mi: array_like`.
  
  One or more mutual information values in nats.
  If this is a Pandas `DataFrame` or `Series`, the columns and indices
  are preserved.



<hr style="margin: 4rem 0;">

# `pairwise_mi`
Estimate the pairwise MI between each variable.

- Unconditional MI: the default.
- Conditional MI: pass a `cond` array.
- Discrete-continuous MI: currently not supported.

Returns a matrix where the $(i,j)$'th element is the mutual information
between the $i$'th and $j$'th columns in the data.
The values are in nats or in the normalized scale depending on the `normalize` parameter.
The diagonal contains NaNs (for better visualization, as the auto-MI should be
$\infty$ nats or correlation $1$).

### Positional or keyword parameters
- `data: array_like`.
  
  A 2D array where the columns represent variables.

### Optional keyword parameters
- `k: int`, default 3.
  
  The number of neighbors to consider.

- `cond: array_like` or None.
  
  Optional 1D or 2D array of observations used for conditioning.
  Must have as many observations as the data.
  A $(n \times m)$ array is interpreted as a single $m$-dimensional variable.

- `mask: array_like` or None.
    
  If specified, an array of booleans that gives the data elements to use for
  estimation. Use this to exclude some observations from consideration.

- `preprocess: bool`, default True.
  
  By default, the variables are scaled to unit variance and
  added with low-amplitude noise. The noise uses a fixed random seed.

- `drop_nan: bool`, default False.

  If True, all NaN (not a number) values are masked out.

- `normalize: bool`, default False.

  If True, the MI values will be normalized to correlation coefficient scale.

- `max_threads: int` or None.
  
  The maximum number of threads to use for estimation.
  By default, the number of CPU cores is used.
  Because the calculation is CPU-bound, more threads should not be used.

- `callback: method` or None.
  
  A method to call when each estimation task is completed. The method
  must take two integer parameters, representing the variable indices.

  This method should be very short. Because Python code is not executed
  concurrently, the callback may slow down other estimation tasks.

  See also: [Example progress reporter](snippets.md).
