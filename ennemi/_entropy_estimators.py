"""The estimator methods.

Do not import this module directly.
Use the `estimate_mi` method in the main ennemi module instead.
"""

import bisect
import math
import numpy as np
from scipy.special import psi

def _estimate_single_mi(x : np.ndarray, y : np.ndarray, k : int = 3):
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

    # We use the fastest O(N*sqrt(k)) time algorithm, based on boxes
    # Create the 2D grid for finding the k-th neighbor
    # TODO Performance: Having specialized 2D and 3D grids would benefit this
    # method by ~10%, but at the expense of duplicated code.
    grid = _BoxGrid(x, y, k)

    # Sort the observation arrays
    # These arrays is used for finding the axis neighbors by binary search
    xs = np.sort(x)
    ys = np.sort(y)

    # Go through all observations
    nx = np.empty(N, np.int32)
    ny = np.empty(N, np.int32)

    for i in range(0, N):
        cur_x = x[i]
        cur_y = y[i]

        eps = _find_kth_neighbor(k, grid, cur_x, cur_y, 0)

        # Now eps contains the distance to the k'th neighbor
        # Find the number of x and y neighbors within that distance
        # This also includes the element itself but that cancels out in the
        # formula of Kraskov et al.
        nx[i] = _count_within(xs, cur_x - eps, cur_x + eps)
        ny[i] = _count_within(ys, cur_y - eps, cur_y + eps)

    # Calculate the estimate
    # TODO: Rounding errors?
    return psi(N) + psi(k) - np.mean(psi(nx) + psi(ny))


def _estimate_conditional_mi(x : np.ndarray, y : np.ndarray, cond : np.ndarray,
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
    grid = _BoxGrid(x, y, k, cond)
    zs = np.sort(cond)

    nxz = np.empty(N, np.int32)
    nyz = np.empty(N, np.int32)
    nz = np.empty(N, np.int32)
    for i in range(0, N):
        cur_x = x[i]
        cur_y = y[i]
        cur_z = cond[i]

        eps = _find_kth_neighbor(k, grid, cur_x, cur_y, cur_z)

        # We can use _count_within only in one case
        nz[i] = _count_within(zs, cur_z - eps, cur_z + eps)

        # The other two terms are 2D projections and there is no well-ordering
        # for 2D vectors. However, using vectorized NumPy operations is faster
        # in benchmarks than applying the BoxGrid data structure. The grid is
        # good for k'th-neighbor search but inefficient for counting.
        nxz[i] = np.sum(np.maximum(np.abs(x - cur_x), np.abs(cond - cur_z)) < eps)
        nyz[i] = np.sum(np.maximum(np.abs(y - cur_y), np.abs(cond - cur_z)) < eps)

    return psi(k) - np.mean(psi(nxz) + psi(nyz) - psi(nz))


def _find_kth_neighbor(k, grid, cur_x, cur_y, cur_z):
    # Start with the box containing the point. Then until we have found
    # enough neighbors, extend the rectangle to the most potential direction.
    # We should not go like peeling an onion because the boxes are not
    # regularly spaced.
    #
    # So, the search order could look like this:
    # +----------+---+-------+-----------+--------+
    # |          |   |       |           |        |
    # |    6     | 6 | 6     |    6      |   7    |
    # +-------------------------------------------+
    # |          |   |       |           |        |
    # |    4     | 3 | 3     |    5      |   7    |
    # +-------------------------------------------+
    # |          |   |       |           |        |
    # |          |   |       |           |        |
    # |    4     | 1 | x 0   |    5      |   7    |
    # |          |   |       |           |        |
    # +-------------------------------------------+
    # |    4     | 2 |  2    |    5      |   7    |
    # +----------+---+-------+-----------+--------+
    # where x is (cur_x, cur_y). Of course, the search would stop as soon
    # as extending further would provide no more benefit.

    eps = float("inf")
    distances = np.full(k, eps)

    # First go through the points in the current box
    box_x, box_y, box_z = grid.find_box(cur_x, cur_y, cur_z)
    eps = _update_epsilon(distances, eps, cur_x, cur_y, cur_z, grid.boxes[box_x, box_y, box_z])

    # Then extend to the most promising direction as long as necessary    
    left = right = box_x
    down = up = box_y
    back = front = box_z
    M = grid.xy_boxes - 1
    Z = grid.z_boxes - 1

    while not (left == 0 and right == M and down == 0 and up == M):

        # Find the direction where the edge is the closest
        edge_dist = np.full(6, np.inf)
        if left > 0: edge_dist[0] = cur_x - grid.x_splits[left]
        if right < M: edge_dist[1] = grid.x_splits[right+1] - cur_x
        if down > 0: edge_dist[2] = cur_y - grid.y_splits[down]
        if up < M: edge_dist[3] = grid.y_splits[up+1] - cur_y
        if back > 0: edge_dist[4] = cur_z - grid.z_splits[back]
        if front < Z: edge_dist[5] = grid.z_splits[front+1] - cur_z

        direction = np.argmin(edge_dist)

        # If all the points in that direction are further away
        # than the points found so far, we are done
        if eps <= edge_dist[direction]:
            return eps

        # Otherwise, extend the rectangle to that direction and go through
        # all the boxes that form the added edge
        if direction == 0:
            # Go left
            left -= 1
            for j in range(down, up+1):
                for k in range(back, front+1):
                    eps = _update_epsilon(distances, eps, cur_x, cur_y, cur_z, grid.boxes[left, j, k])
        elif direction == 1:
            # Go right
            right += 1
            for j in range(down, up+1):
                for k in range(back, front+1):
                    eps = _update_epsilon(distances, eps, cur_x, cur_y, cur_z, grid.boxes[right, j, k])
        elif direction == 2:
            # Go down
            down -= 1
            for i in range(left, right+1):
                for k in range(back, front+1):
                    eps = _update_epsilon(distances, eps, cur_x, cur_y, cur_z, grid.boxes[i, down, k])
        elif direction == 3:
            # Go up
            up += 1
            for i in range(left, right+1):
                for k in range(back, front+1):
                    eps = _update_epsilon(distances, eps, cur_x, cur_y, cur_z, grid.boxes[i, up, k])
        elif direction == 4:
            # Go back
            back -= 1
            for i in range(left, right+1):
                for j in range(down, up+1):
                    eps = _update_epsilon(distances, eps, cur_x, cur_y, cur_z, grid.boxes[i, j, back])
        elif direction == 5:
            # Go forward
            front += 1
            for i in range(left, right+1):
                for j in range(down, up+1):
                    eps = _update_epsilon(distances, eps, cur_x, cur_y, cur_z, grid.boxes[i, j, front])

    # This is reachable if the space contains just one box
    return eps


def _update_epsilon(distances, eps, cur_x, cur_y, cur_z, box):
    for (x, y, z) in box:
        dist = max(abs(cur_x - x), abs(cur_y - y), abs(cur_z - z))

        # Do not count the point itself
        if 0 < dist < eps:
            # Replace the largest distance in the array, then re-sort
            distances[len(distances)-1] = dist
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


class _BoxGrid:
    """A helper for accessing the space in smaller blocks."""

    def __init__(self, x : np.ndarray, y : np.ndarray, k : int, z : np.ndarray = None):
        if z is None:
            # Each box contains k points on average
            # (Benchmarked to be better than 2*k or 4*k)
            split_size = int(math.sqrt(k * len(x)))
            xy_boxes = math.ceil(len(x) / split_size)
            z_boxes = 1
        else:
            # The same rule but in three dimensions
            split_size = int(math.pow(k * len(x) * len(x), 1/3))
            xy_boxes = math.ceil(len(x) / split_size)
            z_boxes = xy_boxes

        # Store the boxes in a dictionary keyed by block coordinates
        self.xy_boxes = xy_boxes
        self.z_boxes = z_boxes
        self.boxes = {}
        for i in range(xy_boxes):
            for j in range(xy_boxes):
                for k in range(z_boxes):
                    self.boxes[(i,j,k)] = []

        # Now the real initialization: first find the split points
        self.x_splits = np.sort(x)[0:len(x):split_size]
        self.y_splits = np.sort(y)[0:len(x):split_size]
        if not z is None:
            self.z_splits = np.sort(z)[0:len(x):split_size]
        else:
            self.z_splits = np.full(1, -np.inf)
        
        # Then assign points to boxes based on those
        for i in range(len(x)):
            if z is None: zi = 0
            else: zi = z[i]

            box_x, box_y, box_z = self.find_box(x[i], y[i], zi)
            self.boxes[(box_x, box_y, box_z)].append((x[i], y[i], zi))


    def find_box(self, x : float, y : float, z : float):
        # The box index to use is the index of the largest split point
        # smaller than the coordinate. This is easy to get with bisect,
        # we just need to offset by one.
        # This algorithm is O(log n).
        box_x = bisect.bisect(self.x_splits, x) - 1
        box_y = bisect.bisect(self.y_splits, y) - 1
        box_z = bisect.bisect(self.z_splits, z) - 1
        return (box_x, box_y, box_z)
