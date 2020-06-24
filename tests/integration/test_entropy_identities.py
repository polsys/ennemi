"""Mathematical identities for entropy and mutual information."""

from ennemi import estimate_entropy, estimate_mi
import numpy as np # type: ignore
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
