"""Tests for ennemi._estimate_single_mi() and friends."""

from math import log
import numpy as np
from scipy.special import psi
import unittest
from ennemi._entropy_estimators import _estimate_single_mi, _estimate_conditional_mi

class TestEstimateSingleMi(unittest.TestCase):

    def test_bivariate_gaussian(self):
        cases = [ (0, 40, 3, 0.1),
                  (0, 200, 3, 0.05),
                  (0, 2000, 3, 0.005),
                  (0, 2000, 5, 0.006),
                  (0, 2000, 20, 0.003),
                  (0.5, 200, 3, 0.05),
                  (0.5, 200, 5, 0.02),
                  (0.5, 2000, 3, 0.02),
                  (-0.9, 200, 3, 0.05),
                  (-0.9, 2000, 3, 0.05),
                  (-0.9, 2000, 5, 0.02), ]
        for (rho, n, k, delta) in cases:
            with self.subTest(rho=rho, n=n, k=k):
                rng = np.random.default_rng(0)
                cov = np.array([[1, rho], [rho, 1]])

                data = rng.multivariate_normal([0, 0], cov, size=n)
                x = data[:,0]
                y = data[:,1]

                actual = _estimate_single_mi(x, y, k=k)
                expected = -0.5 * log(1 - rho**2)
                self.assertAlmostEqual(actual, expected, delta=delta)

    def test_sum_of_exponentials(self):
        # We define X ~ Exp(1), W ~ Exp(2) and Y = X + W.
        # Now by arXiv:1609.02911, Y has known, closed-form entropy.
        cases = [ (1, 2), (0.2, 0.3), (3, 3.1) ]
        for (a, b) in cases:
            with self.subTest(a=a, b=b):
                rng = np.random.default_rng(20200302)
                x = rng.exponential(1/a, 1000)
                w = rng.exponential(1/b, 1000)
                y = x + w

                actual = _estimate_single_mi(x, y, k=5)
                expected = np.euler_gamma + log((b-a)/a) + psi(b/(b-a))

                self.assertAlmostEqual(actual, expected, delta=0.025)

    def test_independent_uniform(self):
        # We have to use independent random numbers instead of linspace,
        # because the algorithm has trouble with evenly spaced values
        rng = np.random.default_rng(1)
        x = rng.uniform(0.0, 1.0, 1024)
        y = rng.uniform(0.0, 1.0, 1024)

        actual = _estimate_single_mi(x, y, k=8)
        actual2 = _estimate_single_mi(y, x, k=8)
        self.assertAlmostEqual(actual, 0, delta=0.04)
        self.assertAlmostEqual(actual, actual2, delta=0.00001)

    def test_independent_transformed_uniform(self):
        # Very non-uniform density, but should have equal MI as the uniform test
        rng = np.random.default_rng(1)
        x = rng.uniform(0.0, 10.0, 1024)
        y = rng.uniform(0.0, 10.0, 1024)

        actual = _estimate_single_mi(x, y, k=8)
        self.assertAlmostEqual(actual, 0, delta=0.04)


class TestEstimateConditionalMi(unittest.TestCase):

    def test_gaussian_with_independent_condition(self):
        # In this case, the results should be same as in ordinary MI
        cases = [ (0.5, 200, 3, 0.03),
                  (0.75, 400, 3, 0.01),
                  (-0.9, 4000, 5, 0.03), ]
        for (rho, n, k, delta) in cases:
            with self.subTest(rho=rho, n=n, k=k):
                rng = np.random.default_rng(0)
                cov = np.array([[1, rho], [rho, 1]])

                data = rng.multivariate_normal([0, 0], cov, size=n)
                x = data[:,0]
                y = data[:,1]
                cond = rng.uniform(0, 1, size=n)

                actual = _estimate_conditional_mi(x, y, cond, k=k)
                expected = -0.5 * log(1 - rho**2)
                self.assertAlmostEqual(actual, expected, delta=delta)

    def test_gaussian_with_condition_equal_to_y(self):
        # MI(X;Y | Y) should be equal to 0
        rng = np.random.default_rng(4)
        cov = np.array([[1, 0.6], [0.6, 1]])

        data = rng.multivariate_normal([0, 0], cov, size=314)
        x = data[:,0]
        y = data[:,1]

        actual = _estimate_conditional_mi(x, y, y, k=4)
        self.assertAlmostEqual(actual, 0.0, delta=0.001)

    def test_three_gaussians(self):
        # First example in doi:10.1103/PhysRevLett.99.204101,
        # we know the analytic expression for conditional MI of a three-
        # dimensional Gaussian random variable. Here, the covariance matrix
        # is not particularly interesting. The expected CMI expression
        # contains determinants of submatrices.
        rng = np.random.default_rng(5)
        cov = np.array([[1, 1, 1], [1, 4, 1], [1, 1, 9]])

        data = rng.multivariate_normal([0, 0, 0], cov, size=1000)

        actual = _estimate_conditional_mi(data[:,0], data[:,1], data[:,2])
        expected = 0.5 * (log(8) + log(35) - log(9) - log(24))
        self.assertAlmostEqual(actual, expected, delta=0.015)

    def test_four_gaussians(self):
        # As above, but now the condition is two-dimensional.
        # The covariance matrix is defined by transforming a standard normal
        # distribution (u1, u2, u3, u4) as follows:
        #   x  = u1,
        #   y  = u2 + u3 + 2*u4,
        #   z1 = 2*u1 + u3,
        #   z2 = u1 + u4.
        # Unconditionally, x and y are independent, but conditionally they aren't.
        rng = np.random.default_rng(25)
        cov = np.array([[1, 0, 2, 1],
                        [0, 6, 1, 2],
                        [2, 1, 5, 2],
                        [1, 2, 2, 2]])

        # The data needs to be normalized for estimation accuracy,
        # and the sample size must be quite large
        data = rng.multivariate_normal([0, 0, 0, 0], cov, size=8000)
        data = data / np.sqrt(np.var(data, axis=0))

        actual = _estimate_conditional_mi(data[:,0], data[:,1], data[:,2:])
        expected = 0.64964
        self.assertAlmostEqual(actual, expected, delta=0.04)


# Test our custom implementation of the digamma function
from ennemi._entropy_estimators import _psi

class TestPsi(unittest.TestCase):

    def test_comparison_with_scipy(self):
        # Zero
        self.assertEqual(_psi(0), np.inf)

        # Small values
        for x in range(1, 20):
            with self.subTest(x=x):
                self.assertAlmostEqual(_psi(x), psi(x), delta=0.0001)

        # Large values
        for x in range(2, 1000):
            with self.subTest(x=10*x):
                self.assertAlmostEqual(_psi(x*10), psi(x*10), delta=0.0001)
