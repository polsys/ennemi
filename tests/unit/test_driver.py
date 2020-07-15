# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

"""Tests for ennemi.estimate_mi()."""

from itertools import product
import math
import numpy as np
import os.path
import pandas as pd
import random
from typing import List
import unittest
from ennemi import estimate_entropy, estimate_mi, normalize_mi, pairwise_mi

X_Y_DIFFERENT_LENGTH_MSG = "x and y must have same length"
X_COND_DIFFERENT_LENGTH_MSG = "x and cond must have same length"
X_WRONG_DIMENSION_MSG = "x must be one- or two-dimensional"
Y_WRONG_DIMENSION_MSG = "y must be one-dimensional"
COND_WRONG_DIMENSION_MSG = "cond must be one- or two-dimensional"
MASK_WRONG_DIMENSION_MSG = "mask must be one-dimensional"
K_TOO_LARGE_MSG = "k must be smaller than number of observations (after lag and mask)"
K_NEGATIVE_MSG = "k must be greater than zero"
TOO_LARGE_LAG_MSG = "lag is too large, no observations left"
INVALID_MASK_LENGTH_MSG = "mask length does not match input length"
INVALID_MASK_TYPE_MSG = "mask must contain only booleans"
NANS_LEFT_MSG = "input contains NaNs (after applying the mask)"


class TestEstimateEntropy(unittest.TestCase):

    def test_input_shorter_than_k(self) -> None:
        with self.assertRaises(ValueError) as cm:
            estimate_entropy(np.zeros(3), k=3)
        self.assertEqual(str(cm.exception), K_TOO_LARGE_MSG)

    def test_k_must_be_positive(self) -> None:
        for k in [-2, 0]:
            with self.subTest(k=k):
                with self.assertRaises(ValueError) as cm:
                    estimate_entropy(np.zeros(20), k=k)
                self.assertEqual(str(cm.exception), K_NEGATIVE_MSG)

    def test_k_must_be_integer(self) -> None:
        with self.assertRaises(TypeError):
            estimate_entropy(np.zeros(20), k=2.71828) # type: ignore

    def test_x_has_wrong_dimension(self) -> None:
        for dim in [(), (20,2,1)]:
            with self.subTest(dim=dim):
                with self.assertRaises(ValueError) as cm:
                    estimate_entropy(np.zeros(dim))
                self.assertEqual(str(cm.exception), X_WRONG_DIMENSION_MSG)

    def test_mask_is_not_boolean(self) -> None:
        with self.assertRaises(TypeError) as cm:
            estimate_entropy(np.zeros(5), mask=[1,2,3,4,5])
        self.assertEqual(str(cm.exception), INVALID_MASK_TYPE_MSG)

    def test_mask_has_wrong_size(self) -> None:
        with self.assertRaises(ValueError) as cm:
            estimate_entropy(np.zeros(5), mask=[True, False])
        self.assertEqual(str(cm.exception), INVALID_MASK_LENGTH_MSG)

    def test_mask_has_wrong_dimension(self) -> None:
        with self.assertRaises(ValueError) as cm:
            estimate_entropy(np.zeros((5,2)), mask=np.full((5,2), True))
        self.assertEqual(str(cm.exception), MASK_WRONG_DIMENSION_MSG)

    def test_mask_leaves_too_few_observations(self) -> None:
        with self.assertRaises(ValueError) as cm:
            estimate_entropy(np.zeros(5), mask=[False, False, False, True, True])
        self.assertEqual(str(cm.exception), K_TOO_LARGE_MSG)

    def test_cond_must_have_same_length_as_x(self) -> None:
        with self.assertRaises(ValueError) as cm:
            estimate_entropy(np.zeros(5), cond=np.zeros(7))
        self.assertEqual(str(cm.exception), X_COND_DIFFERENT_LENGTH_MSG)

    def test_cond_has_wrong_dimension(self) -> None:
        for dim in [(), (20,2,1)]:
            with self.subTest(dim=dim):
                with self.assertRaises(ValueError) as cm:
                    estimate_entropy(np.zeros(20), cond=np.zeros(dim))
                self.assertEqual(str(cm.exception), COND_WRONG_DIMENSION_MSG)

    def test_single_dimensional_variable_as_list(self) -> None:
        rng = np.random.default_rng(0)
        x = [rng.uniform(0, 2) for _ in range(400)]

        result = estimate_entropy(x)

        self.assertEqual(result.shape, ())
        self.assertAlmostEqual(result, math.log(2 - 0), delta=0.01)

    def test_multidim_interpretation(self) -> None:
        # Generate a two-dimensional Gaussian variable
        rng = np.random.default_rng(1)
        cov = np.asarray([[1, 0.6], [0.6, 2]])
        data = rng.multivariate_normal([0, 0], cov, size=1500)

        marginal = estimate_entropy(data)
        multidim = estimate_entropy(data, multidim=True)

        # If multidim=False, we get marginal entropies
        self.assertEqual(marginal.shape, (2,))
        self.assertAlmostEqual(marginal[0],
            0.5*math.log(2*math.pi*math.e*1), delta=0.03)
        self.assertAlmostEqual(marginal[1],
            0.5*math.log(2*math.pi*math.e*2), delta=0.03)

        # If multidim=True, we get the combined entropy
        self.assertEqual(multidim.shape, ())
        self.assertAlmostEqual(multidim.item(),
            math.log(2*math.pi*math.e) + 0.5*math.log(2-0.6**2), delta=0.04)

    def test_pandas_dataframe(self) -> None:
        rng = np.random.default_rng(1)
        data = pd.DataFrame({
            "N": rng.normal(0.0, 1.0, size=500),
            "Unif": rng.uniform(0.0, 0.5, size=500),
            "Exp": rng.exponential(1/2.0, size=500)
        })

        marginal = estimate_entropy(data)
        multidim = estimate_entropy(data, multidim=True)

        # multidim=False results in a DataFrame
        self.assertIsInstance(marginal, pd.DataFrame)
        self.assertEqual(marginal.shape, (1,3))
        self.assertAlmostEqual(marginal.loc[0,"N"], 0.5*math.log(2*math.pi*math.e), delta=0.04)
        self.assertAlmostEqual(marginal.loc[0,"Unif"], math.log(0.5), delta=0.03)
        self.assertAlmostEqual(marginal.loc[0,"Exp"], 1.0 - math.log(2.0), delta=0.07)

        # multidim=True results in a NumPy scalar
        # There is no reference value, the check just guards for regressions
        self.assertEqual(multidim.shape, ())
        self.assertAlmostEqual(multidim.item(), 1.22, delta=0.02)

    def test_pandas_series(self) -> None:
        rng = np.random.default_rng(2)
        data = pd.Series(rng.normal(0.0, 1.0, size=500), name="N")

        result = estimate_entropy(data)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (1,1))
        self.assertAlmostEqual(result.loc[0,"N"], 0.5*math.log(2*math.pi*math.e), delta=0.02)

    def test_nans_must_be_masked(self) -> None:
        rng = np.random.default_rng(3)
        data = rng.normal(0.0, 1.0, size=(800,2))
        data[0:10,0] = np.nan
        data[5:15,1] = np.nan

        # Without masking, the NaNs are rejected
        with self.assertRaises(ValueError) as cm:
            estimate_entropy(data)
        self.assertEqual(str(cm.exception), NANS_LEFT_MSG)

        # With masking, a correct result is produced
        mask = np.full(800, True)
        mask[0:15] = False

        result = estimate_entropy(data, mask=mask)
        expected = 0.5*math.log(2*math.pi*math.e)
        self.assertAlmostEqual(result[0], expected, delta=0.03)
        self.assertAlmostEqual(result[1], expected, delta=0.03)

    def test_conditional_entropy_1d_condition(self) -> None:
        # Draw a sample from three-dimensional Gaussian distribution
        rng = np.random.default_rng(4)
        cov = np.asarray([[1, 0.6, 0.3], [0.6, 2, 0.1], [0.3, 0.1, 1]])
        data = rng.multivariate_normal([0, 0, 0], cov, size=1500)

        marginal = estimate_entropy(data, cond=data[:,2])
        multidim = estimate_entropy(data[:,:2], cond=data[:,2], multidim=True)

        # By the chain rule of entropy, H(X|Y) = H(X,Y) - H(Y)
        def expected(ix: int, iy: int) -> float:
            joint = 0.5 * math.log((2 * math.pi * math.e)**2 * (cov[ix,ix]*cov[iy,iy] - cov[ix,iy]**2))
            single = 0.5 * math.log(2 * math.pi * math.e * cov[iy,iy])
            return joint - single

        self.assertAlmostEqual(marginal[0], expected(0,2), delta=0.04)
        self.assertAlmostEqual(marginal[1], expected(1,2), delta=0.09)
        self.assertLess(marginal[2], -4.5)

        expected_multi = 0.5 * (math.log(np.linalg.det(2 * math.pi * math.e * cov)) - math.log(2*math.pi*math.e*1))
        self.assertAlmostEqual(multidim, expected_multi, delta=0.1)

    def test_conditional_entropy_nd_condition(self) -> None:
        # Draw a sample from three-dimensional Gaussian distribution
        rng = np.random.default_rng(5)
        cov = np.asarray([[1, 0.6, 0.3], [0.6, 2, 0.1], [0.3, 0.1, 1]])
        data = rng.multivariate_normal([0, 0, 0], cov, size=1000)

        marginal = estimate_entropy(data, cond=data)
        multidim = estimate_entropy(data, cond=data, multidim=True)

        # All of the entropies should be -inf, but practically this does not happen
        self.assertLess(marginal[0], -0.5)
        self.assertLess(marginal[1], -0.5)
        self.assertLess(marginal[2], -0.5)
        self.assertLess(multidim, -1.5)

    def test_conditional_entropy_with_independent_condition(self) -> None:
        rng = np.random.default_rng(6)
        data = rng.normal(0.0, 1.0, size=1200)
        cond = rng.uniform(0.0, 1.0, size=1200)

        uncond_result = estimate_entropy(data)
        cond_result = estimate_entropy(data, cond=cond)

        self.assertAlmostEqual(cond_result, uncond_result, delta=0.05)


class TestEstimateMi(unittest.TestCase):
    
    def test_inputs_of_different_length(self) -> None:
        x = np.zeros(10)
        y = np.zeros(20)

        with self.assertRaises(ValueError) as cm:
            estimate_mi(x, y)
        self.assertEqual(str(cm.exception), X_Y_DIFFERENT_LENGTH_MSG)

    def test_inputs_shorter_than_k(self) -> None:
        x = np.zeros(3)
        y = np.zeros(3)

        with self.assertRaises(ValueError) as cm:
            estimate_mi(x, y, k=5)
        self.assertEqual(str(cm.exception), K_TOO_LARGE_MSG)

    def test_continuous_y_must_be_numeric(self) -> None:
        x = np.zeros(3)
        y = np.asarray(["One", "Two", "Three"])

        with self.assertRaises(TypeError):
            estimate_mi(x, y, k=1)

    def test_k_must_be_positive(self) -> None:
        x = np.zeros(30)
        y = np.zeros(30)

        for k in [-2, 0]:
            with self.subTest(k=k):
                with self.assertRaises(ValueError) as cm:
                    estimate_mi(x, y, k=k)
                self.assertEqual(str(cm.exception), K_NEGATIVE_MSG)

    def test_k_must_be_integer(self) -> None:
        x = np.zeros(30)
        y = np.zeros(30)

        with self.assertRaises(TypeError):
            estimate_mi(x, y, k=2.71828) # type: ignore

    def test_lag_leaves_too_few_observations(self) -> None:
        x = np.zeros(30)
        y = np.zeros(30)

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x, lag=[-5, 10], k=15)
        self.assertEqual(str(cm.exception), K_TOO_LARGE_MSG)

    def test_mask_leaves_no_observations(self) -> None:
        x = np.zeros(30)
        y = np.zeros(30)
        mask = np.full(30, False)

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x, mask=mask)
        self.assertEqual(str(cm.exception), K_TOO_LARGE_MSG)

    def test_mask_and_lag_leave_too_few_observations(self) -> None:
        x = np.zeros(30)
        y = np.zeros(30)
        mask = np.full(30, True)
        mask[:15] = False

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x, lag=-5, mask=mask, k=10)
        self.assertEqual(str(cm.exception), K_TOO_LARGE_MSG)

    def test_lag_too_large(self) -> None:
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x, lag=4)
        self.assertEqual(str(cm.exception), TOO_LARGE_LAG_MSG)

    def test_lag_too_small(self) -> None:
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x, lag=-4)
        self.assertEqual(str(cm.exception), TOO_LARGE_LAG_MSG)

    def test_lag_leaves_no_y_observations(self) -> None:
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x, lag=[2, -2])
        self.assertEqual(str(cm.exception), TOO_LARGE_LAG_MSG)

    def test_cond_lag_leaves_no_y_observations(self) -> None:
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x, lag=1, cond=y, cond_lag=4)
        self.assertEqual(str(cm.exception), TOO_LARGE_LAG_MSG)

    def test_lag_not_integer(self) -> None:
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(TypeError):
            estimate_mi(y, x, lag=1.2)

    def test_x_and_cond_different_length(self) -> None:
        x = np.zeros(10)
        y = np.zeros(20)

        with self.assertRaises(ValueError) as cm:
            estimate_mi(x, x, cond=y)
        self.assertEqual(str(cm.exception), X_COND_DIFFERENT_LENGTH_MSG)

    def test_x_with_zero_dimension(self) -> None:
        x = np.zeros(())
        y = np.zeros(10)

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x)
        self.assertEqual(str(cm.exception), X_WRONG_DIMENSION_MSG)

    def test_x_with_too_large_dimension(self) -> None:
        x = np.zeros((10, 2, 3))
        y = np.zeros(10)

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x)
        self.assertEqual(str(cm.exception), X_WRONG_DIMENSION_MSG)

    def test_y_with_wrong_dimension(self) -> None:
        x = np.zeros(10)
        y = np.zeros((10, 2))

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x)
        self.assertEqual(str(cm.exception), Y_WRONG_DIMENSION_MSG)

    def test_mask_with_wrong_dimension(self) -> None:
        x = np.zeros(10)
        y = np.zeros(10)
        mask = np.zeros((10, 2), dtype=np.bool)

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x, mask=mask)
        self.assertEqual(str(cm.exception), MASK_WRONG_DIMENSION_MSG)

    def test_mask_with_wrong_length(self) -> None:
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x, mask = [ False, True ])
        self.assertEqual(str(cm.exception), INVALID_MASK_LENGTH_MSG)

    def test_mask_with_integer_elements(self) -> None:
        # Integer mask leads to difficult to understand subsetting behavior
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]
        mask = [ 3, 2, 1, 0 ]

        with self.assertRaises(TypeError) as cm:
            estimate_mi(y, x, mask=mask)
        self.assertEqual(str(cm.exception), INVALID_MASK_TYPE_MSG)

    def test_mask_with_mixed_element_types(self) -> None:
        # Integer mask leads to difficult to understand subsetting behavior
        x = [1, 2, 3, 4]
        y = [5, 6, 7, 8]
        mask = [ True, 2, 1, 0 ]

        with self.assertRaises(TypeError) as cm:
            estimate_mi(y, x, mask=mask)
        self.assertEqual(str(cm.exception), INVALID_MASK_TYPE_MSG)

    def test_two_covariates_without_lag(self) -> None:
        rng = np.random.default_rng(0)
        x1 = rng.uniform(0, 1, 100)
        x2 = rng.uniform(0, 1, 100)
        y = x1 + rng.normal(0, 0.01, 100)

        actual = estimate_mi(y, np.array([x1, x2]).T)

        self.assertEqual(actual.shape, (1, 2))
        # The first x component is almost equal to y,
        # while the second x component is independent
        self.assertGreaterEqual(actual[0,0], 2.5)
        self.assertAlmostEqual(actual[0,1], 0, delta=0.02)

    def test_one_variable_with_no_lag(self) -> None:
        rng = np.random.default_rng(1)
        xy = rng.uniform(0, 1, 40)

        actual = estimate_mi(xy, xy, lag=[0, 1, -1])

        self.assertEqual(actual.shape, (3, 1))
        # As above, MI of xy with itself is high, but successive values are independent
        self.assertGreaterEqual(actual[0,0], 2.5)
        self.assertAlmostEqual(actual[1,0], 0, delta=0.1)
        self.assertAlmostEqual(actual[2,0], 0, delta=0.1)

    def test_one_variable_with_lag(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 1, 40)
        y = np.zeros(40)
        y[1:] = x[:-1]

        actual = estimate_mi(y, x, lag=[0, 1, -1])

        self.assertEqual(actual.shape, (3, 1))
        self.assertAlmostEqual(actual[0,0], 0, delta=0.1)
        self.assertGreaterEqual(actual[1,0], 2.5)
        self.assertAlmostEqual(actual[2,0], 0, delta=0.1)
        self.assertAlmostEqual(actual[2,0], 0, delta=0.1)

    def test_one_variable_with_single_lag_as_ndarray(self) -> None:
        # There was a bug where ndarray(1) wouldn't be accepted
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 1, 40)
        y = np.zeros(40)
        y[1:] = x[:-1]

        actual = estimate_mi(y, x, lag=np.asarray(1))

        self.assertEqual(actual.shape, (1, 1))
        self.assertGreaterEqual(actual[0,0], 2.5)

    def test_one_variable_with_lead(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 1, 40)
        y = np.zeros(40)
        y[:-1] = x[1:]

        actual = estimate_mi(y, x, lag=[0, 1, -1])

        self.assertEqual(actual.shape, (3, 1))
        self.assertAlmostEqual(actual[0,0], 0, delta=0.1)
        self.assertAlmostEqual(actual[1,0], 0, delta=0.1)
        self.assertAlmostEqual(actual[2,0], math.exp(1), delta=0.02)

    def test_one_variable_with_lists(self) -> None:
        # The parameters are plain Python lists
        rng = random.Random(0)
        x = [rng.uniform(0, 1) for i in range(100)]
        y = [rng.uniform(0, 1) for i in range(100)]

        actual = estimate_mi(y, x)

        self.assertAlmostEqual(actual[0,0], 0, delta=0.05)

    def test_two_variables_with_lists(self) -> None:
        # Plain Python lists, merged into a list of tuples
        rng = random.Random(1)
        x1 = [rng.uniform(0, 1) for i in range(201)]
        x2 = [rng.uniform(0, 1) for i in range(201)]
        y = [rng.uniform(0, 1) for i in range(201)]

        actual = estimate_mi(y, list(zip(x1, x2)))

        self.assertEqual(actual.shape, (1, 2))
        self.assertAlmostEqual(actual[0,0], 0, delta=0.05)
        self.assertAlmostEqual(actual[0,1], 0, delta=0.05)

    def test_array_from_file(self) -> None:
        # A realistic use case
        script_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_path, "example_data.csv")
        data = np.loadtxt(data_path, delimiter=",", skiprows=1)

        for max_threads in [ None, 1, 2 ]:
            with self.subTest(max_threads=max_threads):
                actual = estimate_mi(data[:,0], data[:,1:4], lag=[0, 1, 3], max_threads=max_threads)

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

    def test_pandas_data_frame(self) -> None:
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

    def test_pandas_data_and_mask_with_custom_indices(self) -> None:
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

    def test_mask_without_lag(self) -> None:
        rng = np.random.default_rng(10)
        cov = np.array([[1, 0.8], [0.8, 1]])

        data = rng.multivariate_normal([0, 0], cov, size=1000)
        x = data[:,0]
        y = data[:,1]
        y[150:450] = -5
        mask = np.full(1000, True)
        mask[150:450] = False

        # The unmasked estimation is clearly incorrect
        unmasked = estimate_mi(y, x, mask=None)
        self.assertLess(unmasked, 0.4)

        # Constraining to the correct dataset produces correct results
        expected = -0.5 * math.log(1 - 0.8**2)
        masked = estimate_mi(y, x, mask=mask)
        self.assertAlmostEqual(masked, expected, delta=0.03)

    def test_mask_as_list(self) -> None:
        x = list(range(300)) # type: List[float]
        for i in range(0, 300, 2):
            x[i] = math.nan

        y = list(range(300, 0, -1))
        mask = [ True, False ] * 150

        self.assertGreater(estimate_mi(y, x, lag=1, mask=mask), 4)

    def test_mask_and_lag(self) -> None:
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

    def test_conditional_mi_with_several_lags(self) -> None:
        # X(t) ~ Normal(0, 1), Y(t) = X(t-1) + noise and Z(t) = Y(t-1) + noise.
        # Now MI(Y,lag=1; Z) is greater than zero but MI(Y,lag=1; Z | X,lag=+1)
        # should be equal to zero. On the other hand, MI(Y;Z) = MI(Y;Z|X) = 0
        # when there is no lag at all.
        rng = np.random.default_rng(20200305)
        x = rng.normal(0, 1, size=802)
        y = x[1:801] + rng.normal(0, 0.001, size=800)
        z = x[0:800] + rng.normal(0, 0.001, size=800)
        x = x[2:802]

        # As a consistency check, the non-conditional MI should be large
        noncond = estimate_mi(z, y, k=5, lag=1)
        self.assertGreater(noncond, 1)

        # Then the conditional MI
        actual = estimate_mi(z, y, lag=[0,1], k=5, cond=x, cond_lag=[1, 2])

        self.assertEqual(actual.shape, (2, 1))
        self.assertAlmostEqual(actual[0,0], 0.0, delta=0.05)
        self.assertAlmostEqual(actual[1,0], 0.0, delta=0.01)

    def test_conditional_mi_with_mask_and_lags(self) -> None:
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

    def test_conditional_mi_with_negative_lag(self) -> None:
        # There was a bug where negative max_lag led to an empty array
        # Here Y is dependent on both X and Z with lead of one time step
        rng = np.random.default_rng(15)
        cov = np.array([[1, 1, 1], [1, 4, 1], [1, 1, 9]])
        data = rng.multivariate_normal([0, 0, 0], cov, size=800)

        x = data[:799,0]
        y = data[1:,1]
        z = data[:799,2]

        actual = estimate_mi(y, x, cond=z, lag=-1, cond_lag=-1)
        expected = 0.5 * (math.log(8) + math.log(35) - math.log(9) - math.log(24))

        self.assertAlmostEqual(actual[0,0], expected, delta=0.025)

    def test_conditional_mi_with_pandas(self) -> None:
        script_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_path, "example_data.csv")
        data = pd.read_csv(data_path)

        actual = estimate_mi(data["y"], data["x1"], lag=-1,
                             cond=data["x2"], cond_lag=0)

        self.assertIsInstance(actual, pd.DataFrame)
        self.assertEqual(actual.shape, (1, 1))
        self.assertAlmostEqual(actual.loc[-1,"x1"], 0.0, delta=0.03)

    def test_conditional_mi_with_multidimensional_cond(self) -> None:
        # X, Y, Z are standard normal and W = X+Y+Z.
        # Therefore I(X;W) < I(X;W | Y) < I(X;W | Y,Z).
        rng = np.random.default_rng(16)
        x = rng.normal(size=600)
        y = rng.normal(size=600)
        z = rng.normal(size=600)
        w = x + y + z

        single_cond = estimate_mi(w, x, cond=y)
        many_cond = estimate_mi(w, x, cond=np.asarray([y,z]).T)

        self.assertEqual(many_cond.shape, (1,1))
        self.assertAlmostEqual(single_cond.item(), 0.33, delta=0.03)
        self.assertAlmostEqual(many_cond.item(), 1.12, delta=0.03)

    def test_unmasked_nans_are_rejected(self) -> None:
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

    def test_unmasked_nans_in_discrete_y_are_rejected(self) -> None:
        x = np.zeros(100)
        y = np.zeros(100)
        y[25] = np.nan

        with self.assertRaises(ValueError) as cm:
            estimate_mi(y, x, discrete_y=True)
        self.assertEqual(str(cm.exception), NANS_LEFT_MSG)

    def test_cond_and_mask_as_list(self) -> None:
        x = [1, 2, 3, 4, 5, math.nan]
        y = [2, 4, 6, 8, 10, 12]
        cond = [1, 1, 2, 3, 5, 8]
        mask = [True, True, True, True, True, False]

        # Not checking for the (bogus) result, just that this
        # type-checks and does not crash
        estimate_mi(y, x, cond=cond, mask=mask)


    def test_discrete_y(self) -> None:
        # See the two_disjoint_uniforms algorithm test
        rng = np.random.default_rng(51)
        y = rng.choice([0, 2], size=800)
        x = rng.uniform(y, y+1)

        mi = estimate_mi(y, x, discrete_y=True)
        self.assertAlmostEqual(mi, math.log(2), delta=0.02)

        # If the parameters are put the wrong way, a warning is emitted
        with self.assertWarns(UserWarning):
            _ = estimate_mi(x, y, discrete_y=True)

    def test_discrete_nonnumeric_y(self) -> None:
        rng = np.random.default_rng(52)
        y = rng.choice([0, 2], size=800)
        x = rng.uniform(y, y+1)

        # The discrete variable may be non-numeric
        y = np.asarray(["Zero", "One", "Two"])[y]

        mi = estimate_mi(y, x, discrete_y=True)
        self.assertAlmostEqual(mi, math.log(2), delta=0.02)


    def test_normalization(self) -> None:
        rng = np.random.default_rng(17)
        cov = np.asarray([[1, 0.6], [0.6, 1]])
        data = rng.multivariate_normal([0, 0], cov, 1000)

        result = estimate_mi(data[:,0], data[:,1], normalize=True)

        self.assertAlmostEqual(result, 0.6, delta=0.02)


    def test_callback_on_single_task(self) -> None:
        callback_results = []
        def callback(var_index: int, lag: int) -> None:
            callback_results.append((var_index, lag))

        _ = estimate_mi([1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8], callback=callback)

        self.assertEqual(len(callback_results), 1)
        self.assertIn((0, 0), callback_results)

    def test_callback_on_many_tasks(self) -> None:
        # Use larger N to force multithreading
        for N in [100, 3000]:
            with self.subTest(N=N):
                callback_results = []
                def callback(var_index: int, lag: int) -> None:
                    callback_results.append((var_index, lag))

                rng = np.random.default_rng(18)
                data = rng.multivariate_normal([0] * 11, np.eye(11), N)

                _ = estimate_mi(data[:,0], data[:,1:], lag=[7, -3, 2], callback=callback)

                self.assertEqual(len(callback_results), 30)
                for (i, lag) in product(range(10), [7, -3, 2]):
                    self.assertIn((i, lag), callback_results)


class TestNormalizeMi(unittest.TestCase):

    def test_correct_values(self) -> None:
        for rho in [0.0, 0.01, 0.2, 0.8]:
            with self.subTest(rho=rho):
                # MI for two correlated unit variance Gaussians
                mi = -0.5 * np.log(1 - rho**2)

                self.assertAlmostEqual(normalize_mi(mi), rho, delta=0.001)

    def test_negative_values_are_passed_as_is(self) -> None:
        for value in [-0.01, -0.1, -1, -np.inf]:
            with self.subTest(value=value):
                self.assertAlmostEqual(normalize_mi(value), value, delta=0.001)

    def test_array_is_handled_elementwise(self) -> None:
        mi = np.asarray([[0.1, 0.5], [0, -1]])
        cor = normalize_mi(mi)

        self.assertEqual(cor.shape, (2, 2))
        # Large deltas because I looked these up manually - the order is what matters
        self.assertAlmostEqual(cor[0,0], 0.4, delta=0.05)
        self.assertAlmostEqual(cor[0,1], 0.8, delta=0.05)
        self.assertAlmostEqual(cor[1,0], 0.0, delta=0.001)
        self.assertAlmostEqual(cor[1,1], -1.0, delta=0.001)

    def test_list_is_handled_elementwise(self) -> None:
        mi = [-1, 0, 1]
        cor = normalize_mi(mi)

        self.assertEqual(cor.shape, (3,))
        self.assertAlmostEqual(cor[0], -1.0, delta=0.001)
        self.assertAlmostEqual(cor[1], 0.0, delta=0.001)
        self.assertAlmostEqual(cor[2], 0.93, delta=0.03)

    def test_pandas_structure_is_preserved(self) -> None:
        mi = np.asarray([[0.1, 0.5], [0, -1]])
        mi = pd.DataFrame(mi, columns=["A", "B"], index=[14, 52])

        cor = normalize_mi(mi)
        self.assertAlmostEqual(cor.loc[14,"A"], 0.4, delta=0.05)
        self.assertAlmostEqual(cor.loc[14,"B"], 0.8, delta=0.05)
        self.assertAlmostEqual(cor.loc[52,"A"], 0.0, delta=0.001)
        self.assertAlmostEqual(cor.loc[52,"B"], -1.0, delta=0.001)


class TestPairwiseMi(unittest.TestCase):

    def generate_normal(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        cov = np.asarray([[1, 0.6], [0.6, 1]])
        return rng.multivariate_normal([0, 0], cov, 1000)

    def test_data_has_three_dimensions(self) -> None:
        data = np.full((10, 3, 2), 0.0)
        with self.assertRaises(ValueError) as cm:
            pairwise_mi(data)
        self.assertEqual(str(cm.exception), X_WRONG_DIMENSION_MSG)

    def test_k_larger_than_observations(self) -> None:
        data = np.reshape(np.arange(20), (10,2))
        
        # Without mask
        with self.assertRaises(ValueError) as cm:
            pairwise_mi(data, k=10)
        self.assertEqual(str(cm.exception), K_TOO_LARGE_MSG)

        # With mask
        mask = np.full(10, True)
        mask[:5] = False
        with self.assertRaises(ValueError) as cm:
            pairwise_mi(data, k=5, mask=mask)
        self.assertEqual(str(cm.exception), K_TOO_LARGE_MSG)

    def test_invalid_k(self) -> None:
        data = np.full((10, 2), 0.0)
        with self.assertRaises(ValueError) as cm:
            pairwise_mi(data, k=0)
        self.assertEqual(str(cm.exception), K_NEGATIVE_MSG)

    def test_cond_different_length(self) -> None:
        data = np.full((10, 2), 0.0)
        cond = np.full(9, 0.0)
        
        with self.assertRaises(ValueError) as cm:
            pairwise_mi(data, cond=cond)
        self.assertEqual(str(cm.exception), X_COND_DIFFERENT_LENGTH_MSG)

    def test_ndarray(self) -> None:
        data = self.generate_normal(100)
        expected = -0.5 * math.log(1 - 0.6**2)

        result = pairwise_mi(data)

        self.assertEqual(result.shape, (2,2))
        self.assertTrue(np.isnan(result[0,0]))
        self.assertTrue(np.isnan(result[1,1]))
        self.assertAlmostEqual(result[0,1], expected, delta=0.03)
        self.assertAlmostEqual(result[1,0], expected, delta=0.03)

    def test_pandas(self) -> None:
        rng = np.random.default_rng(101)
        cov = np.asarray([[1, 0.6], [0.6, 1]])
        normal_data = rng.multivariate_normal([0, 0], cov, 1000)
        unif_data = rng.uniform(size=1000)
        expected = -0.5 * math.log(1 - 0.6**2)

        data = pd.DataFrame({"X": normal_data[:,0], "Y": normal_data[:,1], "Z": unif_data})
        result = pairwise_mi(data)

        self.assertEqual(result.shape, (3,3))
        self.assertIsInstance(result, pd.DataFrame)

        for i in "XYZ":
            self.assertTrue(np.isnan(result.loc[i,i]))
        self.assertAlmostEqual(result.loc["X","Y"], expected, delta=0.04)
        self.assertAlmostEqual(result.loc["Y","X"], expected, delta=0.04)
        for i in "XY":
            self.assertAlmostEqual(result.loc[i,"Z"], 0.0, delta=0.03)
            self.assertAlmostEqual(result.loc["Z",i], 0.0, delta=0.03)

    def test_only_one_variable_returns_nan(self) -> None:
        result = pairwise_mi([1, 2, 3, 4])
        
        self.assertEqual(result.shape, (1,1))
        self.assertTrue(np.isnan(result[0,0]))

    def test_only_one_variable_returns_nan_2d_array(self) -> None:
        result = pairwise_mi([[1], [2], [3], [4]])

        self.assertEqual(result.shape, (1,1))
        self.assertTrue(np.isnan(result[0,0]))

    def test_conditioning(self) -> None:
        data = self.generate_normal(102)

        result = pairwise_mi(data, cond=data[:,1])

        self.assertEqual(result.shape, (2,2))
        self.assertTrue(np.isnan(result[0,0]))
        self.assertTrue(np.isnan(result[1,1]))
        self.assertAlmostEqual(result[0,1], 0.0, delta=0.03)
        self.assertAlmostEqual(result[1,0], 0.0, delta=0.03)

    def test_mask_removes_nans(self) -> None:
        data = self.generate_normal(102)
        expected = -0.5 * math.log(1 - 0.6**2)
        data[0:10, 0] = np.nan
        data[5:15, 1] = np.nan

        # Without mask, the estimation should fail
        with self.assertRaises(ValueError) as cm:
            pairwise_mi(data)
        self.assertEqual(str(cm.exception), NANS_LEFT_MSG)

        # With mask, the estimation succeeds
        mask = np.full(1000, True)
        mask[0:15] = False
        result = pairwise_mi(data, mask=mask)

        self.assertEqual(result.shape, (2,2))
        self.assertTrue(np.isnan(result[0,0]))
        self.assertTrue(np.isnan(result[1,1]))
        self.assertAlmostEqual(result[0,1], expected, delta=0.02)
        self.assertAlmostEqual(result[1,0], expected, delta=0.02)

    def test_normalization(self) -> None:
        data = self.generate_normal(104)

        result = pairwise_mi(data, normalize=True)

        self.assertEqual(result.shape, (2,2))
        self.assertTrue(np.isnan(result[0,0]))
        self.assertTrue(np.isnan(result[1,1]))
        self.assertAlmostEqual(result[0,1], 0.6, delta=0.05)
        self.assertAlmostEqual(result[0,1], 0.6, delta=0.05)

    def test_callback(self) -> None:
        # Use larger N to force multithreading
        for N in [100, 3000]:
            with self.subTest(N=N):
                callback_results = []
                def callback(i: int, j: int) -> None:
                    callback_results.append((i, j))

                rng = np.random.default_rng(105)
                data = rng.multivariate_normal([0] * 10, np.eye(10), N)

                _ = pairwise_mi(data, callback=callback)

                self.assertEqual(len(callback_results), 10*9 / 2)
                for (i, j) in product(range(10), range(10)):
                    if i < j:
                        self.assertIn((i, j), callback_results)
                    else:
                        self.assertNotIn((i, j), callback_results)