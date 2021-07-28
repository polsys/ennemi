# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

"""The estimator methods.

Do not import this module directly.
Use the `estimate_mi` method in the main ennemi module instead.
"""

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree
from typing import Union
from warnings import warn

FloatArray = npt.NDArray[np.float64]


def _estimate_single_entropy(x: FloatArray, k: int = 3) -> float:
    """Estimate the differential entropy of a n-dimensional random variable.

    `x` must be a 2D array with columns denoting the variable dimensions.
    1D arrays are promoted to 2D correctly.

    Returns the estimated entropy in nats.
    The calculation is described in Kraskov et al. (2004): Estimating mutual
    information. Physical Review E 69. doi:10.1103/PhysRevE.69.066138
    """

    if x.ndim == 1:
        x = x.reshape((x.size,1))

    N, ndim = x.shape
    grid = cKDTree(x, k)

    # Search for the k'th neighbor of each point and store the distance
    distances = grid.query(x, k=[k+1], p=np.inf)[0].flatten()

    # The log(2) term is because the mean is taken over double the distances
    return _psi(N) - _psi(k) + ndim * (np.mean(np.log(distances)) + np.log(2))


def _estimate_single_mi(x: FloatArray, y: FloatArray, k: int = 3) -> float:
    """Estimate the mutual information between two continuous variables.

    Returns the estimated mutual information (in nats).
    The calculation is based on Kraskov et al. (2004): Estimating mutual
    information. Physical Review E 69. doi:10.1103/PhysRevE.69.066138

    Parameters:
    ---
    x, y : ndarray
        The observed values.
        The two arrays must have the same length.
    k : int
        The number of neighbors to consider. Default 3.
        Must be smaller than the number of observations.

    The algorithm used assumes a continuous distribution. If the data set
    contains many identical observations, this method may return -inf. In that
    case, add low-amplitude noise to the data and try again.
    """

    N = len(x)

    # Ensure that x and y are 2-dimensional
    x = np.column_stack((x,))
    y = np.column_stack((y,))

    # We use the fastest O(N*sqrt(k)) time algorithm
    # Create the 2D tree for finding the k-th neighbor and marginal 1D trees
    xy = np.column_stack((x, y))

    grid = cKDTree(xy)
    x_grid = cKDTree(x)
    y_grid = cKDTree(y)

    # We have to subtract a small value from the radius
    # because the algorithm expects strict inequality but cKDTree also allows equality.
    # This assumes that the radius is of roughly unit magnitude.
    # See https://github.com/polsys/ennemi/issues/76 for justification.
    eps = grid.query(xy, k=[k+1], p=np.inf)[0].flatten()
    nx = x_grid.query_ball_point(x, eps - 1e-12, p=np.inf, return_length=True)
    ny = y_grid.query_ball_point(y, eps - 1e-12, p=np.inf, return_length=True)

    # Calculate the estimate
    return _psi(N) + _psi(k) - np.mean(_psi(nx) + _psi(ny))


def _estimate_conditional_mi(x: FloatArray, y: FloatArray, cond: FloatArray, 
        k: int = 3) -> float:
    """Estimate conditional mutual information between two continuous variables.

    See the documentation for estimate_single_mi for usage.
    The only difference is the additional continuous variable used for
    conditioning.

    The calculation is based on Frenzel & Pompe (2007): Partial Mutual
    Information for Coupling Analysis of Multivariate Time Series.
    Physical Review Letters 99. doi:10.1103/PhysRevLett.99.204101
    """

    # Ensure that cond is 2-dimensional
    cond = np.column_stack((cond,))

    # The cKDTree class offers a lot of vectorization
    # First, create N-dimensional trees for variables
    xyz = np.column_stack((x, y, cond))
    full_grid = cKDTree(xyz)

    xz_grid = cKDTree(np.column_stack((x, cond)))
    yz_grid = cKDTree(np.column_stack((y, cond)))
    z_grid = cKDTree(cond)

    # Find the distance to the k'th neighbor of each point
    eps = full_grid.query(xyz, k=[k+1], p=np.inf)[0].flatten()

    # Find the number of neighbors in marginal spaces
    xz_proj = np.column_stack((x, cond))
    yz_proj = np.column_stack((y, cond))

    # We have to subtract a small value from the radius
    # because the algorithm expects strict inequality but cKDTree also allows equality.
    # This assumes that the radius is of roughly unit magnitude.
    # See https://github.com/polsys/ennemi/issues/76 for justification.
    nxz = xz_grid.query_ball_point(xz_proj, eps - 1e-12, p=np.inf, return_length=True)
    nyz = yz_grid.query_ball_point(yz_proj, eps - 1e-12, p=np.inf, return_length=True)
    nz = z_grid.query_ball_point(cond, eps - 1e-12, p=np.inf, return_length=True)

    return _psi(k) - np.mean(_psi(nxz) + _psi(nyz) - _psi(nz))


def _estimate_semidiscrete_mi(x: FloatArray, y: FloatArray, k: int = 3) -> float:
    """Estimate unconditional MI between discrete y and continuous x.
    
    The calculation is based on Ross (2014): Mutual Information between
    Discrete and Continuous Data Sets. PLoS ONE 9(2):e87357.
    doi:10.1371/journal.pone.0087357
    
    The only difference to basic estimation is that the distance metric
    treats different y values as being further away from each other
    than the marginal distance between any two x values.
    """
    
    N = len(x)

    # Ensure that x is 2-dimensional
    x = np.column_stack((x,))
    
    # Find the unique values of y
    y_values, y_counts = np.unique(y, return_counts=True)

    if len(y_values) > N / 4:
        warn("The discrete variable has relatively many unique values." +
            " Did you pass y and x in correct order?", UserWarning)

    # Create trees for each y value and for the marginal x space
    grids = [cKDTree(x[y==val]) for val in y_values]
    x_grid = cKDTree(x)

    # For each y value:
    # - Find the distance to the k'th neighbor sharing the y value
    # - Find the number of neighbors within that distance in the marginal x space
    # See https://github.com/polsys/ennemi/issues/76 for (eps - 1e-12) tweak.
    n_full = np.empty(N)
    for i, val in enumerate(y_values):
        subset = x[y==val]
        eps = grids[i].query(subset, k=[k+1], p=np.inf)[0].flatten()
        
        n_full[y==val] = x_grid.query_ball_point(subset, eps - 1e-12, p=np.inf, return_length=True)

    # The mean of psi(y_counts) is taken over all sample points, not y buckets
    weighted_y_counts_mean = np.sum(np.dot(_psi(y_counts), y_counts / N))
    return _psi(N) + _psi(k) - np.mean(_psi(n_full)) - weighted_y_counts_mean


def _estimate_conditional_semidiscrete_mi(x: FloatArray, y: FloatArray, cond: FloatArray, 
        k: int = 3) -> float:
    """Estimate conditional MI between discrete y and continuous x and cond.

    This is an adaptation of the CMI algorithm with the 
    discrete-continuous distance metric.
    """

    # Ensure that cond is 2-dimensional
    N = len(y)
    cond = np.column_stack((cond,))

    # Find the unique values of y
    y_values = np.unique(y)
    
    if len(y_values) > N / 4:
        warn("The discrete variable has relatively many unique values." +
            " Did you pass y and x in correct order?", UserWarning)

    # First, create N-dimensional trees for variables
    # The full space is partitioned according to y levels
    xz = np.column_stack((x, cond))
    full_grids = [cKDTree(xz[y==val]) for val in y_values]

    xz_grid = cKDTree(xz)
    z_grid = cKDTree(cond)

    # Similarly, the YZ marginal space is partitioned between y levels
    yz_grids = [cKDTree(cond[y==val]) for val in y_values]

    # Find the distance to the k'th neighbor of each point
    # in the y-partitioned spaces, and find the number of neighbors
    # in marginal spaces.
    xz_proj = np.column_stack((x, cond))
    nxz = np.empty(N)
    nyz = np.empty(N)
    nz = np.empty(N)

    for i, val in enumerate(y_values):
        subset = y==val
        eps = full_grids[i].query(xz[subset], k=[k+1], p=np.inf)[0].flatten()

        # See https://github.com/polsys/ennemi/issues/76 for (eps - 1e-12) tweak.
        nxz[subset] = xz_grid.query_ball_point(xz_proj[subset], eps - 1e-12, p=np.inf, return_length=True)
        nyz[subset] = yz_grids[i].query_ball_point(cond[subset], eps - 1e-12, p=np.inf, return_length=True)
        nz[subset] = z_grid.query_ball_point(cond[subset], eps - 1e-12, p=np.inf, return_length=True)

    return _psi(k) - np.mean(_psi(nxz) + _psi(nyz) - _psi(nz))


#
# Digamma
#

def _psi(x: Union[int, FloatArray]) -> FloatArray:
    """A replacement for scipy.special.psi, for non-negative integers only.
    
    This is slightly faster than the SciPy version (not that it's a bottleneck),
    and has consistent behavior for digamma(0).
    """
    
    x = np.asarray(x)

    # psi(0) = inf for SciPy compatibility
    # The shape of result does not matter as inf will propagate in mean()
    if np.any(x == 0):
        return np.asarray(np.inf)

    # Use the SciPy value for psi(1), because the expansion is not good enough
    mask = (x != 1)
    result = np.full(x.shape, -0.5772156649015331)

    # For the rest, a good enough expansion is given by
    # https://www.uv.es/~bernardo/1976AppStatist.pdf
    y = np.asarray(x[mask], dtype=np.float64)
    result[mask] = np.log(y) - np.power(y, -6) * (np.power(y, 2) * (np.power(y, 2) * (y/2 + 1/12) - 1/120) + 1/252)

    return result
