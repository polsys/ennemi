# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

"""Tests for ennemi._estimate_single_mi() and friends."""

import math
from math import log
import numpy as np
from scipy.special import gamma, psi
import unittest
from ennemi._entropy_estimators import _estimate_single_entropy,\
    _estimate_single_mi, _estimate_conditional_mi, _estimate_semidiscrete_mi


class TestEstimateSingleEntropy(unittest.TestCase):

    def test_univariate_gaussian(self) -> None:
        cases = [ (1, 100, 2, 0.2),
                  (1, 200, 3, 0.05),
                  (1, 2000, 3, 0.02),
                  (0.5, 2000, 3, 0.02),
                  (2.0, 2000, 1, 0.002),
                  (2.0, 2000, 3, 0.02),
                  (2.0, 2000, 30, 0.02), ]
        for (sd, n, k, delta) in cases:
            with self.subTest(sd=sd, n=n, k=k):
                rng = np.random.default_rng(0)
                x = rng.normal(0, sd, size=n)

                actual = _estimate_single_entropy(x, k=k)
                expected = 0.5 * log(2 * math.pi * math.e * sd**2)
                self.assertAlmostEqual(actual, expected, delta=delta)

    def test_uniform(self) -> None:
        cases = [ (0, 1, 1000, 3, 0.05),
                  (1, 2, 1000, 3, 0.05),
                  (-1, 1, 1000, 3, 0.05),
                  (-0.1, 0.1, 1000, 3, 0.05), ]
        for (a, b, n, k, delta) in cases:
            with self.subTest(a=a, b=b, n=n, k=k):
                rng = np.random.default_rng(1)
                x = rng.uniform(a, b, size=n)

                actual = _estimate_single_entropy(x, k=k)
                expected = log(b - a)
                self.assertAlmostEqual(actual, expected, delta=delta)

    def test_bivariate_gaussian(self) -> None:
        cases = [ (0, 1, 200, 3, 0.09),
                  (0, 2, 2000, 3, 0.03),
                  (0.2, 1, 2000, 3, 0.03),
                  (0.2, 2, 2000, 5, 0.03),
                  (0.6, 1, 2000, 1, 0.02),
                  (0.6, 0.5, 2000, 3, 0.04),
                  (0.9, 1, 2000, 3, 0.04),
                  (-0.5, 1, 2000, 5, 0.03), ]
        for (rho, var1, n, k, delta) in cases:
            with self.subTest(rho=rho, var1=var1, n=n, k=k):
                rng = np.random.default_rng(2)
                cov = np.array([[var1, rho], [rho, 1]])
                data = rng.multivariate_normal([0, 0], cov, size=n)

                actual = _estimate_single_entropy(data, k=k)
                expected = 0.5 * log(np.linalg.det(2 * math.pi * math.e * cov))
                self.assertAlmostEqual(actual, expected, delta=delta)

    def test_4d_gaussian(self) -> None:
        rng = np.random.default_rng(3)
        cov = np.array([
            [ 1.0,  0.5,  0.6, -0.2],
            [ 0.5,  1.0,  0.7, -0.5],
            [ 0.6,  0.7,  2.0, -0.1],
            [-0.2, -0.5, -0.1,  0.5]])
        data = rng.multivariate_normal([0, 0, 0, 0], cov, size=2000)

        actual = _estimate_single_entropy(data, k=3)
        expected = 0.5 * log(np.linalg.det(2 * math.pi * math.e * cov))
        self.assertAlmostEqual(actual, expected, delta=0.05)

    def test_gamma_exponential(self) -> None:
        # As in the MI test, the analytical result is due to doi:10.1109/18.825848.
        #
        # x1      ~ Gamma(rate, shape)
        # x2 | x1 ~ Exp(t * x1)
        rng = np.random.default_rng(4)
        r = 1.2
        s = 3.4
        t = 0.56

        x1 = rng.gamma(shape=s, scale=1/r, size=1000)
        x2 = rng.exponential(x1 * t)
        data = np.asarray([x1, x2]).T

        raw = _estimate_single_entropy(data)
        trans = _estimate_single_entropy(np.log(data))

        # The estimate with unlogarithmed data is very bad
        expected = 1 + s - s*psi(s) + log(gamma(s)) - log(t)
        self.assertAlmostEqual(raw, expected, delta=0.65)
        self.assertAlmostEqual(trans, expected, delta=0.01)


class TestEstimateSingleMi(unittest.TestCase):

    def test_bivariate_gaussian(self) -> None:
        cases = [ (0, 40, 3, 0.1),
                  (0, 200, 3, 0.06),
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

    def test_sum_of_exponentials(self) -> None:
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

    def test_independent_uniform(self) -> None:
        # We have to use independent random numbers instead of linspace,
        # because the algorithm has trouble with evenly spaced values
        rng = np.random.default_rng(1)
        x = rng.uniform(0.0, 1.0, 1024)
        y = rng.uniform(0.0, 1.0, 1024)

        actual = _estimate_single_mi(x, y, k=8)
        actual2 = _estimate_single_mi(y, x, k=8)
        self.assertAlmostEqual(actual, 0, delta=0.04)
        self.assertAlmostEqual(actual, actual2, delta=0.00001)

    def test_independent_transformed_uniform(self) -> None:
        # Very non-uniform density, but MI should still be zero
        rng = np.random.default_rng(1)
        x = rng.uniform(0.0, 10.0, 1024)
        y = np.exp(rng.uniform(0.0, 1.0, 1024))

        actual = _estimate_single_mi(x, y, k=8)
        self.assertAlmostEqual(actual, 0, delta=0.02)

    def test_gamma_exponential(self) -> None:
        # Kraskov et al. mention that this distribution is hard to estimate
        # without logarithming the values.
        # The analytical result is due to doi:10.1109/18.825848.
        #
        # x1      ~ Gamma(rate, shape)
        # x2 | x1 ~ Exp(t * x1)
        rng = np.random.default_rng(2)
        r = 1.2
        s = 3.4
        t = 0.56

        x1 = rng.gamma(shape=s, scale=1/r, size=1000)
        x2 = rng.exponential(x1 * t)

        raw = _estimate_single_mi(x1, x2)
        trans = _estimate_single_mi(np.log(x1), np.log(x2))

        expected = psi(s) - np.log(s) + 1/s
        self.assertAlmostEqual(raw, expected, delta=0.04)
        self.assertAlmostEqual(trans, expected, delta=0.005)


class TestEstimateConditionalMi(unittest.TestCase):

    def test_gaussian_with_independent_condition(self) -> None:
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

    def test_gaussian_with_condition_equal_to_y(self) -> None:
        # MI(X;Y | Y) should be equal to 0
        rng = np.random.default_rng(4)
        cov = np.array([[1, 0.6], [0.6, 1]])

        data = rng.multivariate_normal([0, 0], cov, size=314)
        x = data[:,0]
        y = data[:,1]

        actual = _estimate_conditional_mi(x, y, y, k=4)
        self.assertAlmostEqual(actual, 0.0, delta=0.001)

    def test_three_gaussians(self) -> None:
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

    def test_four_gaussians(self) -> None:
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


class TestEstimateSemiDiscreteMi(unittest.TestCase):

    def test_independent_variables(self) -> None:
        cases = [ (2, 200, 3, 0.04),
                  (2, 400, 1, 0.02),
                  (2, 400, 3, 0.02),
                  (2, 800, 8, 0.02),
                  (4, 2000, 2, 0.01) ]
        for (discrete_count, n, k, delta) in cases:
            with self.subTest(count=discrete_count, n=n, k=k):
                rng = np.random.default_rng(50)
                x = rng.normal(0.0, 1.0, size=n)
                y = rng.choice(np.arange(discrete_count), size=n)

                mi = _estimate_semidiscrete_mi(x, y, k)
                self.assertAlmostEqual(max(mi, 0.0), 0.0, delta=delta)

    def test_two_disjoint_uniforms(self) -> None:
        # Y takes two equally probable values, and then X is sampled
        # from two disjoint distributions depending on Y.
        # Therefore I(X;Y) = H(Y) = log(2).
        rng = np.random.default_rng(51)
        y = rng.choice([0, 2], size=800)
        x = rng.uniform(y, y+1)

        mi = _estimate_semidiscrete_mi(x, y)
        self.assertAlmostEqual(mi, log(2), delta=0.02)

    def test_three_disjoint_uniforms(self) -> None:
        # As above, but with three equally probable values for Y.
        rng = np.random.default_rng(51)
        y = rng.choice([0, 2, 5], size=800)
        x = rng.uniform(y, y+1)

        mi = _estimate_semidiscrete_mi(x, y)
        self.assertAlmostEqual(mi, log(3), delta=0.02)

    def test_two_overlapping_uniforms(self) -> None:
        # Here there are two values for Y, but the associated X intervals overlap.
        # Additionally, one of the values is more likely than the other.
        rng = np.random.default_rng(52)
        y = rng.choice([0, 0.7, 0.7], size=2000)
        x = rng.uniform(y, y+1)

        mi = _estimate_semidiscrete_mi(x, y)
        expected = log(3)*7/30 + log(1)*9/30 + log(3/2)*14/30
        self.assertAlmostEqual(mi, expected, delta=0.05)


# Test our custom implementation of the digamma function
from ennemi._entropy_estimators import _psi

class TestPsi(unittest.TestCase):

    def test_comparison_with_scipy(self) -> None:
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
