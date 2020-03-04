"""The estimator methods.

Do not import this module directly, but rather import the main ennemi module.
The `estimate_single_mi` and `estimate_conditional_mi` methods are available
through that module. Prefer using the `estimate_mi` method that combines
the two and provides additional functionality.
"""

import bisect
import numpy as np
from scipy.special import psi

def estimate_single_mi(x : np.ndarray, y : np.ndarray, k : int = 3):
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

    _check_parameters(k, x, y)
    N = len(x)

    # For the initial version, we use the O(N*sqrt(N*k)) time algorithm
    # since it is quite simple to implement. A more complex O(N*sqrt(k))
    # algorithm exists as well (see Kraskov et al.).

    # Rank the observations first by x and then by y
    # This array is used for finding the joint and x neighbors
    inds = np.lexsort((y, x))

    # Sort the observation arrays
    # These arrays is used for finding the axis neighbors by binary search
    xs = np.sort(x)
    ys = np.sort(y)

    # Go through all observations (in the xy sorted order)
    nx = np.empty(N, np.int32)
    ny = np.empty(N, np.int32)

    for i in range(0, N):
        cur_x = x[inds[i]]
        cur_y = y[inds[i]]

        eps = _find_kth_neighbor(k, inds, i, x, y)

        # Now eps contains the distance to the k'th neighbor
        # Find the number of x and y neighbors within that distance
        # This also includes the element itself but that cancels out in the
        # formula of Kraskov et al.
        nx[i] = _count_within(xs, cur_x - eps, cur_x + eps)
        ny[i] = _count_within(ys, cur_y - eps, cur_y + eps)

    # Calculate the estimate
    # TODO: Rounding errors?
    return psi(N) + psi(k) - np.mean(psi(nx) + psi(ny))


def estimate_conditional_mi(x : np.ndarray, y : np.ndarray, cond : np.ndarray,
                            k : int = 3):
    """Estimate conditional mutual information between two continuous variables.

    See the documentation for estimate_single_mi for usage.
    The only difference is the additional continuous variable used for
    conditioning.

    The calculation is based on Frenzel & Pompe (2007): Partial Mutual
    Information for Coupling Analysis of Multivariate Time Series.
    Physical Review Letters 99. doi:10.1103/PhysRevLett.99.204101
    """

    _check_parameters(k, x, y, cond)
    N = len(x)

    # This is a straightforward extension of estimate_single_mi,
    # except that we cannot use _count_within for 2D projections.
    inds = np.lexsort((cond, y, x))
    zs = np.sort(cond)

    nxz = np.empty(N, np.int32)
    nyz = np.empty(N, np.int32)
    nz = np.empty(N, np.int32)
    for i in range(0, N):
        cur_x = x[inds[i]]
        cur_y = y[inds[i]]
        cur_z = cond[inds[i]]

        eps = _find_kth_neighbor(k, inds, i, x, y, cond)

        # We can use _count_within only in one case
        nz[i] = _count_within(zs, cur_z - eps, cur_z + eps)

        # The other two terms are 2D projections and there is no well-ordering
        # for 2D vectors. We'd need a better data structure, but in waiting for
        # that let's at least vectorize the thing properly.
        # TODO: Performance
        nxz[i] = np.sum(np.maximum(np.abs(x - cur_x), np.abs(cond - cur_z)) < eps)
        nyz[i] = np.sum(np.maximum(np.abs(y - cur_y), np.abs(cond - cur_z)) < eps)

    return psi(k) - np.mean(psi(nxz) + psi(nyz) - psi(nz))


def _find_kth_neighbor(k, inds, i, x, y, z=None):
    # Find the coordinates of the k'th neighbor in the joint distribution
    # Do this by first searching to the left in x direction and then to
    # the right in x direction until the x values become too large to
    # qualify for inclusion in k smallest distances.
    #
    # This is not the most efficient method, especially in three dimensions.
    # TODO: Replace with a more performant algorithm
    cur_x = x[inds[i]]
    cur_y = y[inds[i]]
    if not z is None: cur_z = z[inds[i]]

    eps = float("inf")
    z_dist = 0
    distances = np.full(k, eps)
    for j in range(i-1, -1, -1):
        x_dist = abs(cur_x - x[inds[j]])
        if x_dist >= eps:
            break

        y_dist = abs(cur_y - y[inds[j]])
        if not z is None: z_dist = abs(cur_z - z[inds[j]])
        cur_dist = max(x_dist, y_dist, z_dist) # L-infinity norm
        if (cur_dist < eps):
            # Replace the largest distance in the array, then re-sort
            distances[k-1] = cur_dist
            distances.sort()
            eps = distances.max()
    
    for j in range(i+1, len(x)):
        x_dist = abs(cur_x - x[inds[j]])
        if x_dist >= eps:
            break

        y_dist = abs(cur_y - y[inds[j]])
        if not z is None: z_dist = abs(cur_z - z[inds[j]])
        cur_dist = max(x_dist, y_dist, z_dist)
        if (cur_dist < eps):
            distances[k-1] = cur_dist
            distances.sort()
            eps = distances.max()
    
    return eps


def _count_within(array, lower, upper):
    """Returns the number of elements between lower and upper (exclusive).

    The array must be sorted as a binary search will be used.
    This algorithm has O(log n) time complexity.
    """

    # The bisect module provides two methods for adding items to sorted arrays:
    # bisect_left gives insertion point BEFORE duplicates and bisect_right gives
    # insertion point AFTER duplicates. We can just check where the two limits
    # would be added and this gives us our range.
    left = bisect.bisect_right(array, lower)
    right = bisect.bisect_left(array, upper)

    # The interval is strictly between left and right
    # In case of duplicate entries it is possible that right < left
    return max(right - left, 0)


def _check_parameters(k, x, y, cond=None):
    if (len(x) != len(y)):
        raise ValueError("x and y must have same length")
    if (not cond is None) and (len(x) != len(cond)):
        raise ValueError("x and cond must have same length")
    if (len(x) <= k):
        raise ValueError("k must be smaller than number of observations")
