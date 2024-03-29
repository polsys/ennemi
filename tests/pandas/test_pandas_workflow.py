# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

"""A simplified version of the case study in the documentation."""

from __future__ import annotations
from ennemi import estimate_corr, pairwise_corr
import pandas as pd
import unittest
import os

class TestPandasWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        # Import the Kaisaniemi data set used in documentation
        script_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_path, "../../docs/kaisaniemi.csv")

        # Data preprocessing is done by ennemi, the distributions are symmetric
        self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)


    def test_pairwise_corr(self) -> None:
        # Determine the pairwise correlation between three variables
        columns = ["Temperature", "WindDir", "DayOfYear"]
        afternoon_mask = (self.data.index.hour == 13)

        uncond = pairwise_corr(self.data[columns], mask=afternoon_mask) # type: pd.DataFrame
        cond_doy = pairwise_corr(self.data[columns], mask=afternoon_mask,
            cond=self.data["DayOfYear"]) # type: pd.DataFrame

        # The result is a 3x3 data frame
        self.assertEqual(uncond.shape, (3,3))
        self.assertEqual(cond_doy.shape, (3,3))
        self.assertIsInstance(uncond, pd.DataFrame)
        self.assertIsInstance(cond_doy, pd.DataFrame)

        # The matrix is symmetric
        self.assertEqual(uncond.loc["Temperature", "DayOfYear"],
            uncond.loc["DayOfYear", "Temperature"])

        # Temperature is highly dependent on day of year
        self.assertAlmostEqual(uncond.loc["Temperature", "DayOfYear"], 0.9, delta=0.03)

        # There is no correlation with the conditioning variable
        self.assertLess(cond_doy.loc["Temperature", "DayOfYear"], 0.02)

        # The correlation between temperature and wind direction is
        # increased by conditioning on DOY
        self.assertGreater(cond_doy.loc["Temperature", "WindDir"],
            uncond.loc["Temperature", "WindDir"] + 0.1)


    def test_autocorrelation(self) -> None:
        # Determine the autocorrelation of temperature, conditional on DOY
        afternoon_mask = (self.data.index.hour == 13)
        result = estimate_corr(self.data["Temperature"], self.data["Temperature"],
            lag=[0, -24, -10*24], cond=self.data["DayOfYear"],
            mask=afternoon_mask) # type: pd.DataFrame

        # The result is a 3x1 data frame
        self.assertEqual(result.shape, (3,1))
        self.assertIsInstance(result, pd.DataFrame)

        # Without lag, the autocorrelation coefficient should obviously be 1
        self.assertAlmostEqual(result.loc[0, "Temperature"], 1, delta=0.01)

        # With one day lag, the autocorrelation is still very strong
        self.assertAlmostEqual(result.loc[-24, "Temperature"], 0.69, delta=0.01)

        # With ten day lag, the autocorrelation is close to zero
        self.assertAlmostEqual(result.loc[-10*24, "Temperature"], 0, delta=0.01)
