# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

"""A simplified version of the discrete data tutorial in the documentation."""

from __future__ import annotations
from ennemi import estimate_entropy, estimate_mi, pairwise_mi
import numpy as np
import pandas as pd
import unittest
import os

class TestPandasDiscrete(unittest.TestCase):
    def setUp(self) -> None:
        N = 200
        week = np.arange(N)
        rng = np.random.default_rng(1234)

        # A very simple weather model:
        # The weather is completely determined by temperature, air pressure, and wind
        actual_temp = 15 + 5*np.sin(week / 8) + rng.normal(0, 3, N)
        actual_press = 1000 + 30*np.sin(week / 3) + rng.normal(0, 4, N)
        wind_dir = rng.choice([0, 1, 2, 3], N, p=[0.15, 0.15, 0.4, 0.3])

        # Unlike in the tutorial, we convert discrete variables to ints straight away
        # This is an unfortunate limitation with pandas (strings are stored as objects)
        weather = np.full(N, 1)
        weather[(actual_press > 1015) & (actual_temp > 18)] = 2
        weather[(weather==1) & (wind_dir==3)] = 0
        weather[(weather==1) & (wind_dir==2) & rng.choice([0,1], N)] = 0

        self.data = data = pd.DataFrame({
            "Temp": np.round(actual_temp + rng.normal(0, 1, N)),
            "Press": np.round(actual_press + rng.normal(0, 1, N)),
            "Wind": wind_dir,
            "Weather": weather})


    def test_pairwise_mi(self) -> None:
        mi = pairwise_mi(self.data, discrete=[False, False, True, True])

        self.assertAlmostEqual(mi.loc["Weather", "Wind"], 0.533, delta=0.01)
        self.assertAlmostEqual(mi.loc["Weather", "Temp"], 0.092, delta=0.01)


    def test_proportion_of_entropy(self) -> None:
        # Determine the information in the continuous variables
        uncond_continuous = estimate_mi(self.data["Weather"],
            self.data[["Temp", "Press"]], discrete_y=True)
        conditional_press = estimate_mi(self.data["Weather"], self.data["Press"],
            cond=self.data["Temp"], discrete_y=True)

        cont_contribution = uncond_continuous.loc[0, "Temp"] + conditional_press.iloc[0,0]

        # Determine the information in wind direction alone
        wind = estimate_mi(self.data["Weather"], self.data["Wind"],
            discrete_x=True, discrete_y=True)

        # Compute the proportions of information (uncertainty coefficient)
        entropy = estimate_entropy(self.data["Weather"], discrete=True)
        self.assertAlmostEqual(cont_contribution / entropy.iloc[0,0], 0.26, delta=0.005)
        self.assertAlmostEqual(wind.iloc[0,0] / entropy.iloc[0,0], 0.59, delta=0.005)
