# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

"""Verify that the MI and entropy estimates are unbiased."""

from __future__ import annotations
from ennemi import estimate_entropy, estimate_mi
from math import log, pi, e
import numpy as np
import unittest

class TestUnbiasedness(unittest.TestCase):
    
    def test_unconditional_mi_bias(self) -> None:
        # A highly correlated distribution
        rng = np.random.default_rng(0)
        cov = [[1, 0.8], [0.8, 1]]
        data = rng.multivariate_normal([0, 0], cov, size=20_000)
        
        mi_3 = estimate_mi(data[:,0], data[:,1], k=3)
        mi_100 = estimate_mi(data[:,0], data[:,1], k=100)

        # Large k will have some bias, small k should not
        expected = -0.5 * log(1 - 0.8**2)
        self.assertAlmostEqual(mi_3, expected, delta=0.005)
        self.assertGreater(abs(mi_100 - expected), abs(mi_3 - expected) + 0.005)

    def test_unconditional_mi_independence(self) -> None:
        rng = np.random.default_rng(0)
        cov = [[1, 0], [0, 1]]
        data = rng.multivariate_normal([0, 0], cov, size=20_000)
        
        mi_3 = estimate_mi(data[:,0], data[:,1], k=3)
        mi_100 = estimate_mi(data[:,0], data[:,1], k=100)

        # Large k should be better for independence testing
        self.assertAlmostEqual(mi_100, 0.0, delta=0.004)
        self.assertGreater(mi_3 - mi_100, 0.002)


    def test_conditional_mi_independence(self) -> None:
        # X and X+Y are independent given Y
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 1.0, size=20_000)
        y = rng.normal(0.0, 1.0, size=20_000)
        
        mi_3 = estimate_mi(x, x+y, cond=x, k=3)
        mi_100 = estimate_mi(x, x+y, cond=x, k=100)

        # Large k should be better for independence testing here as well
        self.assertAlmostEqual(mi_100, 0.0, delta=0.005)
        self.assertGreater(abs(mi_3 - mi_100), 0.05)


    def test_entropy_bias(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=20_000)

        h_1 = estimate_entropy(x, k=1)
        h_100 = estimate_entropy(x, k=100)

        # Small k has positive bias, large k has negative bias
        expected = 0.5*log(2*pi*e)
        self.assertGreater(h_1, expected + 0.005)
        self.assertLess(h_100, expected - 0.005)

        # Still, both are reasonably close, and large k is closer
        self.assertAlmostEqual(h_1, expected, delta=0.04)
        self.assertAlmostEqual(h_100, expected, delta=0.01)


    def test_conditional_entropy_bias(self) -> None:
        # This is especially interesting as errors might not cancel out in the chain rule
        # Use the 3D Gaussian distribution seen in the driver test
        rng = np.random.default_rng(0)
        cov = np.asarray([[1, 0.6, 0.3], [0.6, 2, 0.1], [0.3, 0.1, 1]])
        data = rng.multivariate_normal([0, 0, 0], cov, size=20_000)

        h_5 = estimate_entropy(data[:,:2], cond=data[:,2], multidim=True, k=5)
        h_50 = estimate_entropy(data[:,:2], cond=data[:,2], multidim=True, k=50)

        # Again, large k appears to have more negative bias
        expected = 0.5 * (log(np.linalg.det(2 * pi * e * cov)) - log(2 * pi * e))
        self.assertAlmostEqual(h_5, expected, delta=0.005)
        self.assertAlmostEqual(h_50, expected, delta=0.03)
