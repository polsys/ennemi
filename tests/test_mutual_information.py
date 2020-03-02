"""Tests for ennemi.estimate_single_mi() and friends."""

from math import log
import numpy as np
from scipy.special import psi
import unittest
from ennemi import estimate_single_mi

class TestEstimateSingleMi(unittest.TestCase):

    def test_inputs_of_different_length(self):
        x = np.zeros(10)
        y = np.zeros(20)

        with self.assertRaises(ValueError):
            estimate_single_mi(x, y)

    def test_inputs_shorter_than_k(self):
        x = np.zeros(3)
        y = np.zeros(3)

        with self.assertRaises(ValueError):
            estimate_single_mi(x, y, k=5)
            estimate_single_mi(x, y)

    def test_k_must_be_positive(self):
        x = np.zeros(30)
        y = np.zeros(30)

        with self.assertRaises(ValueError):
            estimate_single_mi(x, y, k=-2)

    def test_k_must_be_integer(self):
        x = np.zeros(30)
        y = np.zeros(30)

        with self.assertRaises(TypeError):
            estimate_single_mi(x, y, k=2.71828)

    def test_bivariate_gaussian(self):
        cases = [ (0, 20, 3, 0.1),
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

                actual = estimate_single_mi(x, y, k=k)
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

                actual = estimate_single_mi(x, y, k=5)
                expected = np.euler_gamma + log((b-a)/a) + psi(b/(b-a))

                self.assertAlmostEqual(actual, expected, delta=0.025)

    def test_independent_uniform(self):
        # We have to use independent random numbers instead of linspace,
        # because the algorithm has trouble with evenly spaced values
        rng = np.random.default_rng(1)
        x = rng.uniform(0.0, 1.0, 1024)
        y = rng.uniform(0.0, 1.0, 1024)

        actual = estimate_single_mi(x, y, k=8)
        actual2 = estimate_single_mi(y, x, k=8)
        self.assertAlmostEqual(actual, 0, delta=0.04)
        self.assertAlmostEqual(actual, actual2, delta=0.00001)

    def test_independent_transformed_uniform(self):
        # Very non-uniform density, but should have equal MI as the uniform test
        rng = np.random.default_rng(1)
        x = rng.uniform(0.0, 10.0, 1024)
        y = rng.uniform(0.0, 10.0, 1024)

        actual = estimate_single_mi(x, y, k=8)
        self.assertAlmostEqual(actual, 0, delta=0.04)


# Also test the helper method because binary search is easy to get wrong
from ennemi.entropy_estimators import _count_within

class TestCountWithin(unittest.TestCase):

    def test_power_of_two_array(self):
        array = [ 0, 1, 2, 3, 4, 5, 6, 7 ]
        actual = _count_within(array, 2, 5)
        self.assertEqual(actual, 2)

    def test_full_array_within(self):
        array = [ 0, 1, 2, 3, 4, 5 ]
        actual = _count_within(array, -3, 70)
        self.assertEqual(actual, 6)

    def test_left_side(self):
        array = [ 0, 1, 2, 3, 4, 5, 6 ]
        actual = _count_within(array, -1, 2)
        self.assertEqual(actual, 2)

    def test_right_side(self):
        array = [ 0, 1, 2, 3, 4, 5, 6 ]
        actual = _count_within(array, 4, 7)
        self.assertEqual(actual, 2)

    def test_none_matching_at_initial_pos(self):
        array = [ 1, 2, 3, 4 ]
        actual = _count_within(array, 2, 2)
        self.assertEqual(actual, 0)

    def test_none_matching_on_left_side(self):
        array = [ 0.1, 1.2, 2.3, 3.4, 4.5 ]
        actual = _count_within(array, 0.7, 1.1)
        self.assertEqual(actual, 0)

    def test_none_matching_on_right_side(self):
        array = [ 0.1, 1.2, 2.3, 3.4, 4.5 ]
        actual = _count_within(array, 3.4, 4.5)
        self.assertEqual(actual, 0)

    def test_none_matching_in_duplicate_array(self):
        array = [ 1, 1, 1, 1, 1, 1, 1, 1, 1 ]
        actual = _count_within(array, 2, 3)
        self.assertEqual(actual, 0)

    def test_none_matching_in_duplicate_array_and_exact_bounds(self):
        array = [ 1, 1, 1, 1, 1, 1, 1, 1, 1 ]
        actual = _count_within(array, 1, 1)
        self.assertEqual(actual, 0)

    def test_repeated_values(self):
        array = [ 0, 0, 0, 1, 1, 2, 2, 2, 3, 3 ]
        actual = _count_within(array, 0, 3)
        self.assertEqual(actual, 5)
