# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

"""Mathematical identities for entropy and mutual information."""

from __future__ import annotations
from ennemi import estimate_entropy, estimate_mi
import numpy as np
import unittest

class TestEntropyIdentities(unittest.TestCase):
    
    def test_mi_as_conditional_entropy_difference(self) -> None:
        # Make up some kind of distribution
        rng = np.random.default_rng(0)
        x = rng.gamma(shape=2.0, scale=1.0, size=2000)
        y = rng.normal(x, scale=1.0, size=x.shape)

        # We should have I(X;Y) = H(X) - H(X|Y)
        mi = estimate_mi(y, x)
        ent_x = estimate_entropy(x)
        cond_ent = estimate_entropy(x, cond=y)

        self.assertAlmostEqual(ent_x - cond_ent, mi, delta=0.02)
    
    def test_mi_as_sum_of_entropies(self) -> None:
        # Make up another distribution
        rng = np.random.default_rng(1)
        x = rng.chisquare(5, size=8000)
        y = rng.gamma(x, scale=1.0, size=x.shape)

        # We should have I(X;Y) = H(X) + H(Y) - H(X,Y)
        mi = estimate_mi(y, x)
        marginal = estimate_entropy(np.column_stack((x, y)))
        joint = estimate_entropy(np.column_stack((x,y)), multidim=True)

        self.assertAlmostEqual(np.sum(marginal) - joint, mi, delta=0.02)

    def test_discrete_mi_as_conditional_entropy_difference(self) -> None:
        # A -> X, the others are random
        rng = np.random.default_rng(2)
        x = rng.choice(["A", "B", "C", "D"], 200, p=[0.3, 0.1, 0.4, 0.2])
        y = rng.choice(["X", "Y", "Z", "W"], 200, p=[0.1, 0.1, 0.2, 0.6])
        y[x == "A"] = "X"

        mi = estimate_mi(y, x, discrete_y=True, discrete_x=True)
        ent_x = estimate_entropy(x, discrete=True)
        cond_ent = estimate_entropy(x, cond=y, discrete=True)

        self.assertAlmostEqual(ent_x - cond_ent, mi, delta=1e-6)

    def test_discrete_mi_as_sum_of_entropies(self) -> None:
        # A -> X and B -> Y, the others are random
        rng = np.random.default_rng(2)
        x = rng.choice(["A", "B", "C", "D", "E"], 200, p=[0.2, 0.2, 0.3, 0.2, 0.1])
        y = rng.choice(["X", "Y", "Z", "W"], 200, p=[0.1, 0.2, 0.2, 0.5])
        y[x == "A"] = "X"
        y[x == "B"] = "Y"

        mi = estimate_mi(y, x, discrete_y=True, discrete_x=True)
        marginal = estimate_entropy(np.column_stack((x, y)), discrete=True)
        joint = estimate_entropy(np.column_stack((x,y)), multidim=True, discrete=True)

        self.assertAlmostEqual(np.sum(marginal) - joint, mi, delta=1e-6)

