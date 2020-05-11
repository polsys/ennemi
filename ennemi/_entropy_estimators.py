"""The estimator methods.

Do not import this module directly.
Use the `estimate_mi` method in the main ennemi module instead.
"""

import bisect
from typing import Dict, Iterator, List, Tuple, Union
import itertools
import math
import numpy as np # type: ignore


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

    # We use the fastest O(N*sqrt(k)) time algorithm, based on boxes
    # Create the 2D grid for finding the k-th neighbor
    grid = _BoxGrid2D(x, y, k)

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

        eps = _find_kth_neighbor_2d(k, grid, cur_x, cur_y)

        # Now eps contains the distance to the k'th neighbor
        # Find the number of x and y neighbors within that distance
        # This also includes the element itself but that cancels out in the
        # formula of Kraskov et al.
        nx[i] = _count_within_1d(xs, cur_x - eps, cur_x + eps)
        ny[i] = _count_within_1d(ys, cur_y - eps, cur_y + eps)

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

    N = len(x)

    # This is a straightforward extension of estimate_single_mi,
    # except that we use N-dimensional grids everywhere.
    # The boxes are larger than in the unconditional case because of NumPy
    # vectorization. The Z projection has a special case for 1-dimensional
    # condition as the generic code is much slower.
    full_grid = _BoxGridND(np.column_stack((x, y, cond)), 10*k)
    if cond.ndim == 1:
        z_array = np.sort(cond)
    else:
        z_grid = _BoxGridND(np.column_stack((cond,)), 50*k)
    xz_grid = _BoxGridND(np.column_stack((x, cond)), 50*k)
    yz_grid = _BoxGridND(np.column_stack((y, cond)), 50*k)

    nxz = np.empty(N, np.int32)
    nyz = np.empty(N, np.int32)
    nz = np.empty(N, np.int32)

    for i in range(0, N):
        cur_x = x[i]
        cur_y = y[i]
        cur_z = cond[i]

        eps = _find_kth_neighbor_nd(k, full_grid, cur_x, cur_y, cur_z)

        # Count the number of marginal neighbors using the projected grids
        if cond.ndim == 1:
            nz[i] = _count_within_1d(z_array, cur_z - eps, cur_z + eps)
        else:
            nz[i] = _count_within_nd(z_grid, cur_z, eps)
        
        xz_proj = np.concatenate(([cur_x], np.atleast_1d(cur_z)))
        nxz[i] = _count_within_nd(xz_grid, xz_proj, eps)
        yz_proj = np.concatenate(([cur_y], np.atleast_1d(cur_z)))
        nyz[i] = _count_within_nd(yz_grid, yz_proj, eps)

    return _psi(k) - np.mean(_psi(nxz) + _psi(nyz) - _psi(nz))


def _find_kth_neighbor_2d(k: int, grid: "_BoxGrid2D", cur_x: float, cur_y: float) -> float:
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

    eps = np.inf
    distances = np.full(k, eps)

    # First go through the points in the current box
    box_x, box_y = grid.find_box(cur_x, cur_y)
    eps = _update_epsilon_2d(distances, eps, cur_x, cur_y, grid.boxes[box_x, box_y])

    # Then extend to the most promising direction as long as necessary    
    left = right = box_x
    down = up = box_y
    M = grid.xy_boxes - 1

    while not (left == 0 and right == M and down == 0 and up == M):

        # Find the direction where the edge is the closest
        edge_dist = np.full(4, np.inf)
        if left > 0: edge_dist[0] = cur_x - grid.x_splits[left]
        if right < M: edge_dist[1] = grid.x_splits[right+1] - cur_x
        if down > 0: edge_dist[2] = cur_y - grid.y_splits[down]
        if up < M: edge_dist[3] = grid.y_splits[up+1] - cur_y

        direction = edge_dist.argmin()

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
                eps = _update_epsilon_2d(distances, eps, cur_x, cur_y, grid.boxes[left, j])
        elif direction == 1:
            # Go right
            right += 1
            for j in range(down, up+1):
                eps = _update_epsilon_2d(distances, eps, cur_x, cur_y, grid.boxes[right, j])
        elif direction == 2:
            # Go down
            down -= 1
            for i in range(left, right+1):
                eps = _update_epsilon_2d(distances, eps, cur_x, cur_y, grid.boxes[i, down])
        elif direction == 3:
            # Go up
            up += 1
            for i in range(left, right+1):
                eps = _update_epsilon_2d(distances, eps, cur_x, cur_y, grid.boxes[i, up])

    # This is reachable if the space contains just one box
    return eps


def _update_epsilon_2d(distances: np.ndarray, eps: float,
        cur_x: float, cur_y: float, box: "_Box2D") -> float:
    for (x, y) in box:
        dist = max(abs(cur_x - x), abs(cur_y - y))

        # Do not count the point itself
        if 0 < dist < eps:
            # Replace the largest distance in the array, then re-sort
            distances[len(distances)-1] = dist
            distances.sort()
            eps = distances.max()
    
    return eps


def _count_within_1d(array: List[float], lower: float, upper: float) -> int:
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


def _find_kth_neighbor_nd(k: int, grid: "_BoxGridND",
        cur_x: float, cur_y: float, cur_z: Union[float, np.ndarray]) -> float:
    # The same principle as in the 2D case, but with more dimensions.
    # Because the number of dimensions is arbitrary, the code is more generic.
    # The searched box is stored as [min_x, max_x, min_y, max_y, ...] so
    # that even indices are minimum (left, down, ...) and odd maximum coordinates.

    distances = np.full(k, np.inf)

    # First go through the points in the current box
    cur_point = np.concatenate(([cur_x], [cur_y], np.atleast_1d(cur_z)))
    box_coords = grid.find_box(cur_point)
    distances = _update_epsilon_nd(distances, cur_point, grid.boxes[box_coords])

    # Then extend to the most promising direction as long as necessary
    bounds = np.repeat(box_coords, 2)
    M = grid.xyz_boxes - 1
    full_space = np.tile((0, M), grid.ndim)

    while not np.array_equal(bounds, full_space):

        # Find the direction where the edge is the closest
        edge_dist = np.full(2 * grid.ndim, np.inf)
        for i in range(grid.ndim):
            left = bounds[2*i]
            right = bounds[2*i + 1]
            if left > 0: edge_dist[2*i] = cur_point[i] - grid.splits[left,i]
            if right < M: edge_dist[2*i+1] = grid.splits[right+1,i] - cur_point[i]

        direction = edge_dist.argmin()

        # If all the points in that direction are further away
        # than the points found so far, we are done
        if distances[k-1] <= edge_dist[direction]:
            return distances[k-1]

        # Otherwise, extend the rectangle to that direction and go through
        # all the boxes that form the added edge
        if direction % 2 == 0:
            # Even: move "left"
            fixed_point = bounds[direction] - 1
        else:
            # Odd: move "right"
            fixed_point = bounds[direction] + 1
        bounds[direction] = fixed_point

        # Loop through all the boxes on the new face.
        # To do this, we keep the changed coordinate fixed and loop through
        # all others. To do so, we need to build a list of iterators
        # for itertools.product() first.
        iters = [] # type: List[Union[List[int], range]]
        for i in range(grid.ndim):
            if i == direction // 2:
                iters.append([fixed_point])
            else:
                iters.append(range(bounds[i*2], bounds[i*2+1] + 1))
        for coord in itertools.product(*iters):
            distances = _update_epsilon_nd(distances, cur_point, grid.boxes[coord])

    # This is reachable if the space contains just one box
    return distances[k-1]


def _update_epsilon_nd(distances: List[float], cur_point: np.ndarray,
        box: np.ndarray) -> List[float]:
    # The distances array is assumed to be sorted
    point_count = len(distances)
    eps = distances[point_count - 1]

    if box.size == 0:
        return distances
    
    # This is NumPy vectorized
    # Take the points that are closer than eps, but not the point itself
    box_dists = np.max(np.abs(box - cur_point), axis=1)
    box_dists = box_dists[(0 < box_dists) & (box_dists < eps)]

    if box_dists.size == 0:
        return distances

    distances = np.append(distances, box_dists, axis=0)
    distances.sort()
    return distances[:point_count]


def _count_within_nd(grid: "_BoxGridND", center: np.ndarray, eps: float) -> int:
    # Go through all the boxes that may contain neighbors
    min_coord = grid.find_box(center - eps)
    max_coord = grid.find_box(center + eps)

    # Create a list of iterators for itertools.product()
    iters = [range(max(min_coord[i], 0), min(max_coord[i]+1, grid.xyz_boxes)) for i in range(grid.ndim)]

    # Count the close enough points in every box
    # This is vectorized with NumPy and as such the boxes should be large
    result = 0
    for box_coord in itertools.product(*iters):
        points = grid.boxes[box_coord]
        if points.size == 0:
            continue

        dists = np.max(np.abs(points - center), axis=1)
        result += np.sum(dists < eps)
    return result


#
# Grid types
#

_Box2D = List[Tuple[float, float]]

class _BoxGrid2D:
    """A helper for accessing a two-dimensional space in smaller blocks."""

    def __init__(self, x: np.ndarray, y: np.ndarray, k: int):
        # Each box contains k points on average
        # (Benchmarked to be better than 2*k or 4*k)
        split_size = int(math.sqrt(k * len(x)))
        xy_boxes = math.ceil(len(x) / split_size)

        # Store the boxes in a dictionary keyed by block coordinates
        self.xy_boxes = xy_boxes
        self.boxes = {} # type: Dict[Tuple[int, int], _Box2D]
        for i in range(xy_boxes):
            for j in range(xy_boxes):
                    self.boxes[(i,j)] = []

        # Now the real initialization: first find the split points
        self.x_splits = np.sort(x)[0:len(x):split_size]
        self.y_splits = np.sort(y)[0:len(x):split_size]
        
        # Then assign points to boxes based on those
        for i in range(len(x)):
            box_x, box_y = self.find_box(x[i], y[i])
            self.boxes[(box_x, box_y)].append((x[i], y[i]))


    def find_box(self, x: float, y: float) -> Tuple[int, int]:
        # The box index to use is the index of the largest split point
        # smaller than the coordinate. This is easy to get with bisect,
        # we just need to offset by one.
        # This algorithm is O(log n).
        box_x = bisect.bisect(self.x_splits, x) - 1
        box_y = bisect.bisect(self.y_splits, y) - 1
        return (box_x, box_y)


class _BoxGridND:
    """A helper for accessing an N-dimensional space in smaller blocks."""

    def __init__(self, points: np.ndarray, k: int):
        nobs, ndim = points.shape
        self.ndim = ndim

        # For each box to contain k points on average, the marginal
        # splits must contain N/k points on average
        split_size = int(math.pow(nobs, (ndim-1)/ndim) * math.pow(k, 1/ndim))
        xyz_boxes = math.ceil(nobs / split_size)

        # Store the boxes in a dictionary keyed by block coordinates
        self.xyz_boxes = xyz_boxes
        self.boxes = {} # type: Dict[Tuple[int, ...], np.ndarray]
        for coord in itertools.product(range(xyz_boxes), repeat=ndim):
            self.boxes[coord] = []

        # Now the real initialization: first find the split points
        self.splits = np.empty((xyz_boxes, ndim))
        for i in range(ndim):
            self.splits[:,i] = np.sort(points[:,i])[0:nobs:split_size]
        
        # Then assign points to boxes based on those
        for i in range(nobs):
            coord = self.find_box(points[i])
            self.boxes[coord].append(points[i])

        # Convert all the boxes to ndarrays
        for key in self.boxes:
            self.boxes[key] = np.asarray(self.boxes[key])


    def find_box(self, point: np.ndarray) -> Tuple[int, ...]:
        # The box index to use is the index of the largest split point
        # smaller than the coordinate. This is easy to get with bisect,
        # we just need to offset by one.
        # This algorithm is O(log n).
        coord = []
        point = np.atleast_1d(point)
        for i in range(self.ndim):
            coord.append(bisect.bisect(self.splits[:,i], point[i]) - 1)
        return tuple(coord)


#
# Digamma
#

def _psi(x: np.ndarray) -> np.ndarray:
    """A replacement for scipy.special.psi, for non-negative integers only.
    
    This is up to a few times slower than the SciPy version, but it's not fun
    to depend on full SciPy just for this method (even though SciPy is often
    installed with NumPy). This method is not the bottleneck anyways, so the
    difference vanishes in the measurement noise.
    """
    
    x = np.asarray(x)

    # psi(0) = inf for SciPy compatibility
    result = np.full(x.shape, np.inf)
    mask = (x != 0)

    # Use the SciPy value for psi(1), because the expansion is not good enough
    one_mask = (x == 1)
    result[one_mask] = -0.5772156649015331
    mask = mask & ~one_mask

    # For the rest, a good enough expansion is given by
    # https://www.uv.es/~bernardo/1976AppStatist.pdf
    y = np.asarray(x[mask], dtype=np.float64)
    result[mask] = np.log(y) - np.power(y, -6) * (np.power(y, 2) * (np.power(y, 2) * (y/2 + 1/12) - 1/120) + 1/252)

    return result
