# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

"""The estimator methods.

Do not import this module directly.
Use the `estimate_mi` method in the main ennemi module instead.
"""

import numpy as np
from scipy.spatial import cKDTree


def _estimate_single_entropy_1d(x: np.ndarray, k: int = 3) -> float:
    """Estimate the differential entropy of a one-dimensional random variable.

    Returns the estimated entropy in nats.
    The calculation is described in Kraskov et al. (2004): Estimating mutual
    information. Physical Review E 69. doi:10.1103/PhysRevE.69.066138
    """

    N = x.shape[0]
    xs = np.sort(x)
    distances = np.empty(N)

    # Search for the k'th neighbor of each point and store the distance
    for i in range(N):
        left = i - 1
        right = i + 1
        eps = np.inf
        for _ in range(k):
            if left >= 0: ldist = np.abs(xs[i] - xs[left])
            else: ldist = np.inf
            if right < N: rdist = np.abs(xs[i] - xs[right])
            else: rdist = np.inf

            if ldist < rdist:
                eps = ldist
                left -= 1
            else:
                eps = rdist
                right += 1
        distances[i] = eps

    # The log(2) term is because the mean is taken over double the distances
    return _psi(N) - _psi(k) + np.mean(np.log(distances)) + np.log(2)


def _estimate_single_entropy_nd(x: np.ndarray, k: int = 3) -> float:
    """Estimate the differential entropy of a n-dimensional random variable.

    `x` must be a 2D array with columns denoting the variable dimensions.

    Returns the estimated entropy in nats.
    The calculation is described in Kraskov et al. (2004): Estimating mutual
    information. Physical Review E 69. doi:10.1103/PhysRevE.69.066138
    """

    N, ndim = x.shape
    grid = cKDTree(x, k)
    distances = np.empty(N)

    # Search for the k'th neighbor of each point and store the distance
    for i in range(N):
        point = x[i]
        
        distances[i] = np.max(grid.query(point, k=k+1, p=np.inf)[0])

    # The log(2) term is because the mean is taken over double the distances
    return _psi(N) - _psi(k) + ndim * (np.mean(np.log(distances)) + np.log(2))


def _estimate_single_mi(x: np.ndarray, y: np.ndarray, k: int = 3) -> float:
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

    # Ensure that x and y are is 2-dimensional
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
    # TODO: Try to get Kraskov et al. estimator 2 working.
    eps = grid.query(xy, k=[k+1], p=np.inf)[0].flatten()
    nx = x_grid.query_ball_point(x, eps - 1e-12, p=np.inf, return_length=True)
    ny = y_grid.query_ball_point(y, eps - 1e-12, p=np.inf, return_length=True)

    # Calculate the estimate
    return _psi(N) + _psi(k) - np.mean(_psi(nx) + _psi(ny))


def _estimate_conditional_mi(x: np.ndarray, y: np.ndarray, cond: np.ndarray, 
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
    # TODO: Try to get Kraskov et al. estimator 2 adapted to this case.
    nxz = xz_grid.query_ball_point(xz_proj, eps - 1e-12, p=np.inf, return_length=True)
    nyz = yz_grid.query_ball_point(yz_proj, eps - 1e-12, p=np.inf, return_length=True)
    nz = z_grid.query_ball_point(cond, eps - 1e-12, p=np.inf, return_length=True)

    return _psi(k) - np.mean(_psi(nxz) + _psi(nyz) - _psi(nz))

#
# Digamma
#

def _psi(x: np.ndarray) -> np.ndarray:
    """A replacement for scipy.special.psi, for non-negative integers only.
    
    This is slightly faster than the SciPy version (not that it's a bottleneck),
    and has consistent behavior for digamma(0).
    """
    
    x = np.asarray(x)

    # psi(0) = inf for SciPy compatibility
    # The shape of result does not matter as inf will propagate in mean()
    if np.any(x == 0):
        return np.inf

    # Use the SciPy value for psi(1), because the expansion is not good enough
    mask = (x != 1)
    result = np.full(x.shape, -0.5772156649015331)

    # For the rest, a good enough expansion is given by
    # https://www.uv.es/~bernardo/1976AppStatist.pdf
    y = np.asarray(x[mask], dtype=np.float64)
    result[mask] = np.log(y) - np.power(y, -6) * (np.power(y, 2) * (np.power(y, 2) * (y/2 + 1/12) - 1/120) + 1/252)

    return result
