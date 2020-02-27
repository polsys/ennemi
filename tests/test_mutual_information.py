"""Tests for ennemi.estimate_mi()."""

import math
import numpy as np
import unittest
from ennemi import estimate_mi

class TestEstimateMi(unittest.TestCase):

    def test_inputs_of_different_length(self):
        x = np.zeros(10)
        y = np.zeros(20)

        with self.assertRaises(ValueError):
            estimate_mi(x, y)

    def test_inputs_shorter_than_k(self):
        x = np.zeros(3)
        y = np.zeros(3)

        with self.assertRaises(ValueError):
            estimate_mi(x, y, k=5)
            estimate_mi(x, y)

    def test_k_must_be_positive(self):
        x = np.zeros(30)
        y = np.zeros(30)

        with self.assertRaises(ValueError):
            estimate_mi(x, y, k=-2)

    def test_k_must_be_integer(self):
        x = np.zeros(30)
        y = np.zeros(30)

        with self.assertRaises(TypeError):
            estimate_mi(x, y, k=2.71828)

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
                np.random.seed(0)
                cov = np.array([[1, rho], [rho, 1]])

                data = np.random.multivariate_normal([0, 0], cov, size=n)
                x = data[:,0]
                y = data[:,1]

                actual = estimate_mi(x, y, k=k)
                expected = -0.5 * math.log(1 - rho**2)
                self.assertAlmostEqual(actual, expected, delta=delta)

    def test_independent_uniform(self):
        # We have to use independent random numbers instead of linspace,
        # because the algorithm has trouble with evenly spaced values
        np.random.seed(1)
        x = np.random.uniform(0.0, 1.0, 1024)
        y = np.random.uniform(0.0, 1.0, 1024)

        actual = estimate_mi(x, y, k=8)
        actual2 = estimate_mi(y, x, k=8)
        self.assertAlmostEqual(actual, 0, delta=0.02)
        self.assertAlmostEqual(actual, actual2, delta=0.00001)

    def test_independent_transformed_uniform(self):
        # Very non-uniform density, but should have equal MI as the uniform test
        np.random.seed(1)
        x = np.random.uniform(0.0, 10.0, 1024)
        y = np.random.uniform(0.0, 10.0, 1024)

        actual = estimate_mi(x, y, k=8)
        self.assertAlmostEqual(actual, 0, delta=0.02)


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
