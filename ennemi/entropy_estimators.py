"""The estimator methods.

Do not import this module directly, but rather import the main ennemi module.
The `estimate_single_mi` method is available through that module.
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

    N = len(x)

    if (N != len(y)):
        raise ValueError("x and y must have same length")
    if (N <= k):
        raise ValueError("k must be smaller than number of observations")

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

        # Find the coordinates of the k'th neighbor in the joint distribution
        # Do this by first searching to the left in x direction and then to
        # the right in x direction until the x values become too large to
        # qualify for inclusion in k smallest distances.
        eps = float("inf")
        distances = np.full(k, eps)
        for j in range(i-1, -1, -1):
            x_dist = abs(cur_x - x[inds[j]])
            if x_dist >= eps:
                break

            y_dist = abs(cur_y - y[inds[j]])
            cur_dist = max(x_dist, y_dist) # L-infinity norm
            if (cur_dist < eps):
                # Replace the largest distance in the array, then re-sort
                distances[k-1] = cur_dist
                distances.sort()
                eps = distances.max()
        
        for j in range(i+1, N):
            x_dist = abs(cur_x - x[inds[j]])
            if x_dist >= eps:
                break

            y_dist = abs(cur_y - y[inds[j]])
            cur_dist = max(x_dist, y_dist)
            if (cur_dist < eps):
                distances[k-1] = cur_dist
                distances.sort()
                eps = distances.max()

        # Now eps contains the distance to the k'th neighbor
        # Find the number of x and y neighbors within that distance
        # This also includes the element itself but that cancels out in the
        # formula of Kraskov et al.
        nx[i] = _count_within(xs, cur_x - eps, cur_x + eps)
        ny[i] = _count_within(ys, cur_y - eps, cur_y + eps)

    # Calculate the estimate
    # TODO: Rounding errors?
    return psi(N) + psi(k) - np.mean(psi(nx) + psi(ny))

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
