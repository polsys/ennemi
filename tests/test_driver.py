"""Tests for ennemi.estimate_mi()."""

import math
import numpy as np
import os.path
import random
import unittest
from ennemi import estimate_mi

TOO_LARGE_LAG_MSG = "lag is too large, no observations left"

class TestEstimateMi(unittest.TestCase):

    def test_lag_too_large(self):
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(ValueError) as cm:
            estimate_mi(x, y, time_lag=4)
        self.assertEqual(str(cm.exception), TOO_LARGE_LAG_MSG)

    def test_lag_too_small(self):
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(ValueError) as cm:
            estimate_mi(x, y, time_lag=-4)
        self.assertEqual(str(cm.exception), TOO_LARGE_LAG_MSG)

    def test_lag_leaves_no_y_observations(self):
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(ValueError) as cm:
            estimate_mi(x, y, time_lag=[2, -2])
        self.assertEqual(str(cm.exception), TOO_LARGE_LAG_MSG)

    def test_cond_lag_leaves_no_y_observations(self):
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(ValueError) as cm:
            estimate_mi(x, y, time_lag=1, cond=y, cond_lag=3)
        self.assertEqual(str(cm.exception), TOO_LARGE_LAG_MSG)

    def test_lag_not_integer(self):
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(TypeError):
            estimate_mi(x, y, time_lag=1.2)

    def test_two_covariates_without_lag(self):
        rng = np.random.default_rng(0)
        x1 = rng.uniform(0, 1, 100)
        x2 = rng.uniform(0, 1, 100)
        y = x1 + rng.normal(0, 0.01, 100)

        actual = estimate_mi(np.array([x1, x2]), y)

        self.assertEqual(actual.shape, (2, 1))
        # The first x component is almost equal to y and has entropy ~ exp(1),
        # while the second x component is independent
        self.assertAlmostEqual(actual[0,0], math.exp(1), delta=0.1)
        self.assertAlmostEqual(actual[1,0], 0, delta=0.02)

    def test_one_variable_with_no_lag(self):
        rng = np.random.default_rng(1)
        xy = rng.uniform(0, 1, 40)

        actual = estimate_mi(xy, xy, time_lag=[0, 1, -1])

        self.assertEqual(actual.shape, (1, 3))
        # As above, entropy of xy is exp(1), and values are independent.
        self.assertAlmostEqual(actual[0,0], math.exp(1), delta=0.02)
        self.assertAlmostEqual(actual[0,1], 0, delta=0.1)
        self.assertAlmostEqual(actual[0,2], 0, delta=0.1)

    def test_one_variable_with_lag(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 1, 40)
        y = np.zeros(40)
        y[1:] = x[:-1]

        actual = estimate_mi(x, y, time_lag=[0, 1, -1])

        self.assertEqual(actual.shape, (1, 3))
        self.assertAlmostEqual(actual[0,0], 0, delta=0.1)
        self.assertAlmostEqual(actual[0,1], math.exp(1), delta=0.02)
        self.assertAlmostEqual(actual[0,2], 0, delta=0.1)

    def test_one_variable_with_lead(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 1, 40)
        y = np.zeros(40)
        y[:-1] = x[1:]

        actual = estimate_mi(x, y, time_lag=[0, 1, -1])

        self.assertEqual(actual.shape, (1, 3))
        self.assertAlmostEqual(actual[0,0], 0, delta=0.1)
        self.assertAlmostEqual(actual[0,1], 0, delta=0.1)
        self.assertAlmostEqual(actual[0,2], math.exp(1), delta=0.02)

    def test_one_variable_with_lists(self):
        # The parameters are plain Python lists
        rng = random.Random(0)
        x = [rng.uniform(0, 1) for i in range(100)]
        y = [rng.uniform(0, 1) for i in range(100)]

        actual = estimate_mi(x, y)

        self.assertAlmostEqual(actual[0,0], 0, delta=0.05)

    def test_two_variables_with_lists(self):
        # Plain Python lists
        rng = random.Random(1)
        x1 = [rng.uniform(0, 1) for i in range(201)]
        x2 = [rng.uniform(0, 1) for i in range(201)]
        y = [rng.uniform(0, 1) for i in range(201)]

        actual = estimate_mi([x1, x2], y)

        self.assertEqual(actual.shape, (2, 1))
        self.assertAlmostEqual(actual[0,0], 0, delta=0.05)
        self.assertAlmostEqual(actual[1,0], 0, delta=0.05)

    def test_array_from_file(self):
        # A realistic use case
        script_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_path, "example_data.csv")
        data = np.loadtxt(data_path, delimiter=",", skiprows=1, unpack=True)

        actual = estimate_mi(data[1:4], data[0], time_lag=[0, 1, 3])

        # y(t) depends on x1(t+1)
        self.assertAlmostEqual(actual[0,0], 0.0, delta=0.05)
        self.assertGreater(actual[0,1], 0.5)
        self.assertAlmostEqual(actual[0,2], 0.0, delta=0.05)

        # y(t) is completely independent of x2
        for i in range(3):
            self.assertAlmostEqual(actual[1,i], 0.0, delta=0.05)
        
        # y(t) depends on abs(x3(t+3))
        self.assertAlmostEqual(actual[2,0], 0.0, delta=0.07)
        self.assertAlmostEqual(actual[2,1], 0.0, delta=0.05)
        self.assertGreater(actual[2,2], 0.15)

    def test_conditional_mi_with_several_lags(self):
        # X(t) ~ Normal(0, 1), Y(t) = X(t-1) + noise and Z(t) = Y(t-1) + noise.
        # Now MI(Y,lag=1; Z) is greater than zero but MI(Y,lag=1; Z | X,lag=+1)
        # should be equal to zero. On the other hand, MI(Y;Z) = MI(Y;Z|X) = 0
        # when there is no lag at all.
        rng = np.random.default_rng(20200305)
        x = rng.normal(0, 1, size=802)
        y = x[1:801] + rng.normal(0, 0.001, size=800)
        z = x[0:800] + rng.normal(0, 0.001, size=800)
        x = x[2:802]

        # As a sanity check, test the non-conditional MI
        noncond = estimate_mi(y, z, k=5, time_lag=1)
        self.assertGreater(noncond, 1)

        # Then the conditional MI
        actual = estimate_mi(y, z, time_lag=[0,1], k=5, cond=x, cond_lag=1)

        self.assertEqual(actual.shape, (1, 2))
        self.assertAlmostEqual(actual[0,0], 0.0, delta=0.05)
        self.assertAlmostEqual(actual[0,1], 0.0, delta=0.01)
