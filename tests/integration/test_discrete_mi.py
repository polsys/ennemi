# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

"""Reproduce the (Ross 2014) results of discrete-continuous MI.

The setup and reference results are obtained from the supplementary material
in doi:10.1371/journal.pone.0087357, a set of MATLAB scripts.
The results here were produced on unmodified scripts and R2019b Update 4.
"""

from __future__ import annotations
from ennemi import estimate_mi
import numpy as np
import unittest

class TestDiscreteMi(unittest.TestCase):

    def test_square_wave(self) -> None:
        # Create 10000 samples from three square waves
        # The three waves have unequal probabilities
        rng = np.random.default_rng(2020_07_14)
        y = rng.choice([0, 1, 2], p=[0.2/1.7, 1.0/1.7, 0.5/1.7], size=10000)
        x = np.empty(10000)
        x[y==0] = rng.uniform(0.0, 1.0, np.sum(y==0))
        x[y==1] = rng.uniform(0.1, 1.2, np.sum(y==1))
        x[y==2] = rng.uniform(0.2, 1.3, np.sum(y==2))

        # Calculate the (unnormalized) MI
        actual = estimate_mi(y, x, discrete_y=True)

        # Follows from the definition; this is the output from Ross' script
        # converted from bits to nats
        expected = 0.14587 * np.log(2)
        self.assertAlmostEqual(actual, expected, delta=0.01)

    def test_gaussian(self) -> None:
        # Create 10000 samples from three Gaussian distributions
        rng = np.random.default_rng(2020_07_15)
        y = rng.choice([0, 1, 2], p=[0.2/1.7, 1.0/1.7, 0.5/1.7], size=10000)
        x = np.empty(10000)
        x[y==0] = rng.normal(0.4, 0.20, np.sum(y==0))
        x[y==1] = rng.normal(0.5, 0.30, np.sum(y==1))
        x[y==2] = rng.normal(0.8, 0.25, np.sum(y==2))

        # Calculate the (unnormalized) MI
        actual = estimate_mi(y, x, discrete_y=True)

        # As above
        expected = 0.20524 * np.log(2)
        self.assertAlmostEqual(actual, expected, delta=0.01)
