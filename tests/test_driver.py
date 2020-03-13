"""Tests for ennemi.estimate_mi()."""

import math
import numpy as np
import os.path
import random
import unittest
from ennemi import estimate_mi

TOO_LARGE_LAG_MSG = "lag is too large, no observations left"
INVALID_MASK_LENGTH_MSG = "mask length does not match y length"
INVALID_MASK_TYPE_MSG = "mask must contain only booleans"

class TestEstimateMi(unittest.TestCase):

    def test_lag_too_large(self):
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x, time_lag=4)
        self.assertEqual(str(cm.exception), TOO_LARGE_LAG_MSG)

    def test_lag_too_small(self):
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x, time_lag=-4)
        self.assertEqual(str(cm.exception), TOO_LARGE_LAG_MSG)

    def test_lag_leaves_no_y_observations(self):
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x, time_lag=[2, -2])
        self.assertEqual(str(cm.exception), TOO_LARGE_LAG_MSG)

    def test_cond_lag_leaves_no_y_observations(self):
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x, time_lag=1, cond=y, cond_lag=3)
        self.assertEqual(str(cm.exception), TOO_LARGE_LAG_MSG)

    def test_lag_not_integer(self):
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(TypeError):
            estimate_mi(y, x, time_lag=1.2)

    def test_mask_with_wrong_length(self):
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x, mask = [ False, True ])
        self.assertEqual(str(cm.exception), INVALID_MASK_LENGTH_MSG)

    def test_mask_with_integer_elements(self):
        # Integer mask lead to difficult to understand subsetting behavior
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]
        mask = [ 3, 2, 1, 0 ]

        with self.assertRaises(TypeError) as cm:
            estimate_mi(y, x, mask=mask)
        self.assertEqual(str(cm.exception), INVALID_MASK_TYPE_MSG)

    def test_mask_with_mixed_element_types(self):
        # Integer mask lead to difficult to understand subsetting behavior
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]
        mask = [ True, 2, 1, 0 ]

        with self.assertRaises(TypeError) as cm:
            estimate_mi(y, x, mask=mask)
        self.assertEqual(str(cm.exception), INVALID_MASK_TYPE_MSG)

    def test_unknown_parallel_parameter(self):
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(ValueError):
            estimate_mi(y, x, parallel="whatever")

    def test_two_covariates_without_lag(self):
        rng = np.random.default_rng(0)
        x1 = rng.uniform(0, 1, 100)
        x2 = rng.uniform(0, 1, 100)
        y = x1 + rng.normal(0, 0.01, 100)

        actual = estimate_mi(y, np.array([x1, x2]))

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

        actual = estimate_mi(y, x, time_lag=[0, 1, -1])

        self.assertEqual(actual.shape, (1, 3))
        self.assertAlmostEqual(actual[0,0], 0, delta=0.1)
        self.assertAlmostEqual(actual[0,1], math.exp(1), delta=0.02)
        self.assertAlmostEqual(actual[0,2], 0, delta=0.1)

    def test_one_variable_with_lead(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 1, 40)
        y = np.zeros(40)
        y[:-1] = x[1:]

        actual = estimate_mi(y, x, time_lag=[0, 1, -1])

        self.assertEqual(actual.shape, (1, 3))
        self.assertAlmostEqual(actual[0,0], 0, delta=0.1)
        self.assertAlmostEqual(actual[0,1], 0, delta=0.1)
        self.assertAlmostEqual(actual[0,2], math.exp(1), delta=0.02)

    def test_one_variable_with_lists(self):
        # The parameters are plain Python lists
        rng = random.Random(0)
        x = [rng.uniform(0, 1) for i in range(100)]
        y = [rng.uniform(0, 1) for i in range(100)]

        actual = estimate_mi(y, x)

        self.assertAlmostEqual(actual[0,0], 0, delta=0.05)

    def test_two_variables_with_lists(self):
        # Plain Python lists
        rng = random.Random(1)
        x1 = [rng.uniform(0, 1) for i in range(201)]
        x2 = [rng.uniform(0, 1) for i in range(201)]
        y = [rng.uniform(0, 1) for i in range(201)]

        actual = estimate_mi(y, [x1, x2])

        self.assertEqual(actual.shape, (2, 1))
        self.assertAlmostEqual(actual[0,0], 0, delta=0.05)
        self.assertAlmostEqual(actual[1,0], 0, delta=0.05)

    def test_array_from_file(self):
        # A realistic use case
        script_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_path, "example_data.csv")
        data = np.loadtxt(data_path, delimiter=",", skiprows=1, unpack=True)

        parallel_modes = [ None, "always", "disable" ]
        for parallel in parallel_modes:
            with self.subTest(parallel=parallel):
                actual = estimate_mi(data[0], data[1:4], time_lag=[0, 1, 3])

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

    def test_mask_without_lag(self):
        rng = np.random.default_rng(10)
        cov = np.array([[1, 0.8], [0.8, 1]])

        data = rng.multivariate_normal([0, 0], cov, size=1000)
        x = data[:,0]
        y = data[:,1]
        y[150:450] = -5
        mask = np.full(1000, True)
        mask[150:450] = False

        # Sanity check: the unmasked estimation is incorrect
        unmasked = estimate_mi(y, x, mask=None)
        self.assertLess(unmasked, 0.4)

        expected = -0.5 * math.log(1 - 0.8**2)
        masked = estimate_mi(y, x, mask=mask)
        self.assertAlmostEqual(masked, expected, delta=0.03)

    def test_mask_as_list(self):
        x = list(range(300))
        for i in range(0, 300, 2):
            x[i] = math.nan

        y = list(range(300, 0, -1))
        mask = [ True, False ] * 150

        self.assertGreater(estimate_mi(y, x, time_lag=1, mask=mask), 4)

    def test_mask_and_lag(self):
        # Only even y and odd x elements are preserved
        mask = np.arange(0, 1000) % 2 == 0

        rng = np.random.default_rng(11)
        x = rng.normal(0, 1, size=1001)
        y = x[:1000] + rng.normal(0, 0.01, size=1000)
        x = x[1:1001]

        x[mask] = 0
        y[np.logical_not(mask)] = 0

        actual = estimate_mi(y, x, time_lag=1, mask=mask)
        self.assertGreater(actual, 4)

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
        noncond = estimate_mi(z, y, k=5, time_lag=1)
        self.assertGreater(noncond, 1)

        # Then the conditional MI
        actual = estimate_mi(z, y, time_lag=[0,1], k=5, cond=x, cond_lag=1)

        self.assertEqual(actual.shape, (1, 2))
        self.assertAlmostEqual(actual[0,0], 0.0, delta=0.05)
        self.assertAlmostEqual(actual[0,1], 0.0, delta=0.01)

    def test_conditional_mi_with_mask_and_lags(self):
        # This is TestEstimateConditionalMi.test_three_gaussians(),
        # but with Z lagged by 2 and half of the observations deleted.
        rng = np.random.default_rng(12)
        cov = np.array([[1, 1, 1], [1, 4, 1], [1, 1, 9]])

        data = rng.multivariate_normal([0, 0, 0], cov, size=2002)
        mask = np.arange(2000) % 2 == 0
        x = data[2:,0][mask]
        y = data[2:,1][mask]
        z = data[:2000,2][mask]

        lags = [ 0, -1 ]

        actual = estimate_mi(y, x, time_lag=lags, cond=z, cond_lag=2)
        expected = 0.5 * (math.log(8) + math.log(35) - math.log(9) - math.log(24))

        self.assertAlmostEqual(actual[0,0], expected, delta=0.03)
        self.assertAlmostEqual(actual[0,1], 0.0, delta=0.01)
