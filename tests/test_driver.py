"""Tests for ennemi.estimate_mi()."""

import math
import numpy as np
import os.path
import pandas as pd
import random
import unittest
from ennemi import estimate_mi, normalize_mi

X_Y_DIFFERENT_LENGTH_MSG = "x and y must have same length"
X_COND_DIFFERENT_LENGTH_MSG = "x and cond must have same length"
X_WRONG_DIMENSION_MSG = "x must be one- or two-dimensional"
Y_WRONG_DIMENSION_MSG = "y must be one-dimensional"
MASK_WRONG_DIMENSION_MSG = "mask must be one-dimensional"
K_TOO_LARGE_MSG = "k must be smaller than number of observations"
K_NEGATIVE_MSG = "k must be greater than zero"
TOO_LARGE_LAG_MSG = "lag is too large, no observations left"
INVALID_MASK_LENGTH_MSG = "mask length does not match y length"
INVALID_MASK_TYPE_MSG = "mask must contain only booleans"
NANS_LEFT_MSG = "input contains NaNs (after applying the mask)"

class TestEstimateMi(unittest.TestCase):
    
    def test_inputs_of_different_length(self):
        x = np.zeros(10)
        y = np.zeros(20)

        with self.assertRaises(ValueError) as cm:
            estimate_mi(x, y)
        self.assertEqual(str(cm.exception), X_Y_DIFFERENT_LENGTH_MSG)

    def test_inputs_shorter_than_k(self):
        x = np.zeros(3)
        y = np.zeros(3)

        with self.assertRaises(ValueError) as cm:
            estimate_mi(x, y, k=5)
        self.assertEqual(str(cm.exception), K_TOO_LARGE_MSG)

    def test_k_must_be_positive(self):
        x = np.zeros(30)
        y = np.zeros(30)

        for k in [-2, 0]:
            with self.subTest(k=k):
                with self.assertRaises(ValueError) as cm:
                    estimate_mi(x, y, k=k)
                self.assertEqual(str(cm.exception), K_NEGATIVE_MSG)

    def test_k_must_be_integer(self):
        x = np.zeros(30)
        y = np.zeros(30)

        with self.assertRaises(TypeError):
            estimate_mi(x, y, k=2.71828)

    def test_lag_too_large(self):
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x, lag=4)
        self.assertEqual(str(cm.exception), TOO_LARGE_LAG_MSG)

    def test_lag_too_small(self):
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x, lag=-4)
        self.assertEqual(str(cm.exception), TOO_LARGE_LAG_MSG)

    def test_lag_leaves_no_y_observations(self):
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x, lag=[2, -2])
        self.assertEqual(str(cm.exception), TOO_LARGE_LAG_MSG)

    def test_cond_lag_leaves_no_y_observations(self):
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x, lag=1, cond=y, cond_lag=4)
        self.assertEqual(str(cm.exception), TOO_LARGE_LAG_MSG)

    def test_lag_not_integer(self):
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(TypeError):
            estimate_mi(y, x, lag=1.2)

    def test_x_and_cond_different_length(self):
        x = np.zeros(10)
        y = np.zeros(20)

        with self.assertRaises(ValueError) as cm:
            estimate_mi(x, x, cond=y)
        self.assertEqual(str(cm.exception), X_COND_DIFFERENT_LENGTH_MSG)

    def test_x_with_wrong_dimension(self):
        x = np.zeros((10, 2, 3))
        y = np.zeros(10)

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x)
        self.assertEqual(str(cm.exception), X_WRONG_DIMENSION_MSG)

    def test_y_with_wrong_dimension(self):
        x = np.zeros(10)
        y = np.zeros((10, 2))

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x)
        self.assertEqual(str(cm.exception), Y_WRONG_DIMENSION_MSG)

    def test_mask_with_wrong_dimension(self):
        x = np.zeros(10)
        y = np.zeros(10)
        mask = np.zeros((10, 2), dtype=np.bool)

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x, mask=mask)
        self.assertEqual(str(cm.exception), MASK_WRONG_DIMENSION_MSG)

    def test_mask_with_wrong_length(self):
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x, mask = [ False, True ])
        self.assertEqual(str(cm.exception), INVALID_MASK_LENGTH_MSG)

    def test_mask_with_integer_elements(self):
        # Integer mask leads to difficult to understand subsetting behavior
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

        actual = estimate_mi(y, np.array([x1, x2]).T)

        self.assertEqual(actual.shape, (1, 2))
        # The first x component is almost equal to y and has entropy ~ exp(1),
        # while the second x component is independent
        self.assertAlmostEqual(actual[0,0], math.exp(1), delta=0.1)
        self.assertAlmostEqual(actual[0,1], 0, delta=0.02)

    def test_one_variable_with_no_lag(self):
        rng = np.random.default_rng(1)
        xy = rng.uniform(0, 1, 40)

        actual = estimate_mi(xy, xy, lag=[0, 1, -1])

        self.assertEqual(actual.shape, (3, 1))
        # As above, entropy of xy is exp(1), and values are independent.
        self.assertAlmostEqual(actual[0,0], math.exp(1), delta=0.02)
        self.assertAlmostEqual(actual[1,0], 0, delta=0.1)
        self.assertAlmostEqual(actual[2,0], 0, delta=0.1)

    def test_one_variable_with_lag(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 1, 40)
        y = np.zeros(40)
        y[1:] = x[:-1]

        actual = estimate_mi(y, x, lag=[0, 1, -1])

        self.assertEqual(actual.shape, (3, 1))
        self.assertAlmostEqual(actual[0,0], 0, delta=0.1)
        self.assertAlmostEqual(actual[1,0], math.exp(1), delta=0.02)
        self.assertAlmostEqual(actual[2,0], 0, delta=0.1)
        self.assertAlmostEqual(actual[2,0], 0, delta=0.1)

    def test_one_variable_with_single_lag_as_ndarray(self):
        # There was a bug where ndarray(1) wouldn't be accepted
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 1, 40)
        y = np.zeros(40)
        y[1:] = x[:-1]

        actual = estimate_mi(y, x, lag=np.asarray(1))

        self.assertEqual(actual.shape, (1, 1))
        self.assertAlmostEqual(actual[0,0], math.exp(1), delta=0.02)

    def test_one_variable_with_lead(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 1, 40)
        y = np.zeros(40)
        y[:-1] = x[1:]

        actual = estimate_mi(y, x, lag=[0, 1, -1])

        self.assertEqual(actual.shape, (3, 1))
        self.assertAlmostEqual(actual[0,0], 0, delta=0.1)
        self.assertAlmostEqual(actual[1,0], 0, delta=0.1)
        self.assertAlmostEqual(actual[2,0], math.exp(1), delta=0.02)

    def test_one_variable_with_lists(self):
        # The parameters are plain Python lists
        rng = random.Random(0)
        x = [rng.uniform(0, 1) for i in range(100)]
        y = [rng.uniform(0, 1) for i in range(100)]

        actual = estimate_mi(y, x)

        self.assertAlmostEqual(actual[0,0], 0, delta=0.05)

    def test_two_variables_with_lists(self):
        # Plain Python lists, merged into a list of tuples
        rng = random.Random(1)
        x1 = [rng.uniform(0, 1) for i in range(201)]
        x2 = [rng.uniform(0, 1) for i in range(201)]
        y = [rng.uniform(0, 1) for i in range(201)]

        actual = estimate_mi(y, list(zip(x1, x2)))

        self.assertEqual(actual.shape, (1, 2))
        self.assertAlmostEqual(actual[0,0], 0, delta=0.05)
        self.assertAlmostEqual(actual[0,1], 0, delta=0.05)

    def test_array_from_file(self):
        # A realistic use case
        script_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_path, "example_data.csv")
        data = np.loadtxt(data_path, delimiter=",", skiprows=1)

        parallel_modes = [ None, "always", "disable" ]
        for parallel in parallel_modes:
            with self.subTest(parallel=parallel):
                actual = estimate_mi(data[:,0], data[:,1:4], lag=[0, 1, 3], parallel=parallel)

                # The returned object is a plain ndarray
                self.assertIsInstance(actual, np.ndarray)

                # y(t) depends on x1(t+1)
                self.assertAlmostEqual(actual[0,0], 0.0, delta=0.05)
                self.assertGreater(actual[1,0], 0.5)
                self.assertAlmostEqual(actual[2,0], 0.0, delta=0.05)

                # y(t) is completely independent of x2
                for i in range(3):
                    self.assertAlmostEqual(actual[i,1], 0.0, delta=0.05)
                
                # y(t) depends on abs(x3(t+3))
                self.assertAlmostEqual(actual[0,2], 0.0, delta=0.07)
                self.assertAlmostEqual(actual[1,2], 0.0, delta=0.05)
                self.assertGreater(actual[2,2], 0.15)

    def test_pandas_data_frame(self):
        # Same data as in test_array_from_file()
        script_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_path, "example_data.csv")
        data = pd.read_csv(data_path)

        actual = estimate_mi(data["y"], data[["x1", "x2", "x3"]], lag=[0, 1, 3])

        # The returned object is a Pandas data frame, with row and column names!
        self.assertIsInstance(actual, pd.DataFrame)

        # y(t) depends on x1(t+1)
        self.assertAlmostEqual(actual.loc[0,"x1"], 0.0, delta=0.05)
        self.assertGreater(actual.loc[1,"x1"], 0.5)
        self.assertAlmostEqual(actual.loc[3,"x1"], 0.0, delta=0.05)

        # y(t) is completely independent of x2
        for i in [0, 1, 3]:
            self.assertAlmostEqual(actual.loc[i,"x2"], 0.0, delta=0.05)
        
        # y(t) depends on abs(x3(t+3))
        self.assertAlmostEqual(actual.loc[0,"x3"], 0.0, delta=0.07)
        self.assertAlmostEqual(actual.loc[1,"x3"], 0.0, delta=0.05)
        self.assertGreater(actual.loc[3,"x3"], 0.15)

    def test_pandas_data_and_mask_with_custom_indices(self):
        rng = np.random.default_rng(20)
        cov = np.array([[1, 0.8], [0.8, 1]])

        data = rng.multivariate_normal([0, 0], cov, size=1000)
        x = data[:,0]
        y = data[:,1]
        y[150:450] = np.nan
        mask = np.full(1000, True)
        mask[150:450] = False

        df = pd.DataFrame({"x": x, "y": y, "mask": mask},
                          index=range(2000, 3000))

        expected = -0.5 * math.log(1 - 0.8**2)
        masked = estimate_mi(df["y"], df["x"], mask=df["mask"])
        self.assertAlmostEqual(masked.loc[0,"x"], expected, delta=0.03)

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

        self.assertGreater(estimate_mi(y, x, lag=1, mask=mask), 4)

    def test_mask_and_lag(self):
        # Only even y and odd x elements are preserved
        mask = np.arange(0, 1000) % 2 == 0

        rng = np.random.default_rng(11)
        x = rng.normal(0, 1, size=1001)
        y = x[:1000] + rng.normal(0, 0.01, size=1000)
        x = x[1:1001]

        x[mask] = 0
        y[np.logical_not(mask)] = 0

        actual = estimate_mi(y, x, lag=1, mask=mask)
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
        noncond = estimate_mi(z, y, k=5, lag=1)
        self.assertGreater(noncond, 1)

        # Then the conditional MI
        actual = estimate_mi(z, y, lag=[0,1], k=5, cond=x, cond_lag=[1, 2])

        self.assertEqual(actual.shape, (2, 1))
        self.assertAlmostEqual(actual[0,0], 0.0, delta=0.05)
        self.assertAlmostEqual(actual[1,0], 0.0, delta=0.01)

    def test_conditional_mi_with_mask_and_lags(self):
        # This is TestEstimateConditionalMi.test_three_gaussians(),
        # but with Z lagged by 2 and most of the observations deleted.
        rng = np.random.default_rng(12)
        cov = np.array([[1, 1, 1], [1, 4, 1], [1, 1, 9]])

        data = rng.multivariate_normal([0, 0, 0], cov, size=2002)
        mask = np.arange(2000) % 5 == 0

        x = np.zeros(2000)
        y = np.zeros(2000)
        z = np.zeros(2000)
        x[mask] = data[2:,0][mask]
        y[mask] = data[2:,1][mask]
        z[mask] = data[:2000,2][mask]

        lags = [ 0, -1 ]

        actual = estimate_mi(y, x, lag=lags, cond=z, cond_lag=[2, 1], mask=mask)
        expected = 0.5 * (math.log(8) + math.log(35) - math.log(9) - math.log(24))

        self.assertAlmostEqual(actual[0,0], expected, delta=0.01)
        self.assertAlmostEqual(actual[1,0], 0.0, delta=0.01)

    def test_conditional_mi_with_pandas(self):
        script_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_path, "example_data.csv")
        data = pd.read_csv(data_path)

        actual = estimate_mi(data["y"], data["x1"], lag=-1,
                             cond=data["x2"], cond_lag=0)

        self.assertIsInstance(actual, pd.DataFrame)
        self.assertEqual(actual.shape, (1, 1))
        self.assertAlmostEqual(actual.loc[-1,"x1"], 0.0, delta=0.03)

    def test_unmasked_nans_are_rejected(self):
        for (xnan, ynan, condnan) in [(True, False, None),
                                      (False, True, None),
                                      (True, False, False),
                                      (False, True, False),
                                      (False, False, True)]:
            with self.subTest(xnan=xnan, ynan=ynan, condnan=condnan):
                x = np.zeros(100)
                if xnan: x[17] = np.nan

                y = np.zeros(100)
                if ynan: y[25] = np.nan

                if condnan is not None:
                    cond = np.zeros(100)
                    if condnan: cond[37] = np.nan
                else:
                    cond = None

                with self.assertRaises(ValueError) as cm:
                    estimate_mi(y, x, cond=cond)
                self.assertEqual(str(cm.exception), NANS_LEFT_MSG)


class TestNormalizeMi(unittest.TestCase):

    def test_correct_values(self):
        for rho in [0.0, 0.01, 0.2, 0.8]:
            with self.subTest(rho=rho):
                # MI for two correlated unit variance Gaussians
                mi = -0.5 * np.log(1 - rho**2)

                self.assertAlmostEqual(normalize_mi(mi), rho, delta=0.001)

    def test_negative_values_are_passed_as_is(self):
        for value in [-0.01, -0.1, -1, -np.inf]:
            with self.subTest(value=value):
                self.assertAlmostEqual(normalize_mi(value), value, delta=0.001)

    def test_array_is_handled_elementwise(self):
        mi = np.asarray([[0.1, 0.5], [0, -1]])
        cor = normalize_mi(mi)

        self.assertEqual(cor.shape, (2, 2))
        # Large deltas because I looked these up manually - the order is what matters
        self.assertAlmostEqual(cor[0,0], 0.4, delta=0.05)
        self.assertAlmostEqual(cor[0,1], 0.8, delta=0.05)
        self.assertAlmostEqual(cor[1,0], 0.0, delta=0.001)
        self.assertAlmostEqual(cor[1,1], -1.0, delta=0.001)

    def test_list_is_handled_elementwise(self):
        mi = [-1, 0, 1]
        cor = normalize_mi(mi)

        self.assertEqual(cor.shape, (3,))
        self.assertAlmostEqual(cor[0], -1.0, delta=0.001)
        self.assertAlmostEqual(cor[1], 0.0, delta=0.001)
        self.assertAlmostEqual(cor[2], 0.93, delta=0.03)

    def test_pandas_structure_is_preserved(self):
        mi = np.asarray([[0.1, 0.5], [0, -1]])
        mi = pd.DataFrame(mi, columns=["A", "B"], index=[14, 52])

        cor = normalize_mi(mi)
        self.assertAlmostEqual(cor.loc[14,"A"], 0.4, delta=0.05)
        self.assertAlmostEqual(cor.loc[14,"B"], 0.8, delta=0.05)
        self.assertAlmostEqual(cor.loc[52,"A"], 0.0, delta=0.001)
        self.assertAlmostEqual(cor.loc[52,"B"], -1.0, delta=0.001)
