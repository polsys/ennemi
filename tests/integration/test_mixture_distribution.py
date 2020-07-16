# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

"""A mixture distribution that has no analytical expression for MI."""

from ennemi import estimate_mi
from scipy.integrate import dblquad
from scipy.stats import norm, multivariate_normal as mvnorm
import numpy as np
import unittest

class TestMixtureDistribution(unittest.TestCase):
    def setUp(self) -> None:
        # A mixture of two normal distributions
        self.cov1 = np.asarray([[1, 0.6], [0.6, 1]])
        self.cov2 = np.asarray([[1, -0.4], [-0.4, 1]])

        self.mean1 = np.asarray([-1, -1])
        self.mean2 = np.asarray([2, 0.5])


    def test_mi(self) -> None:
        # Estimate the actual MI by numerical integration
        expected = self.integrate_mi()

        # Create random samples from the distribution
        rng = np.random.default_rng(0)
        small_k1 = []
        small_k3 = []
        full_k2 = []
        full_k40 = []

        for _ in range(5):
            full_sample = np.concatenate((
                rng.multivariate_normal(self.mean1, self.cov1, 2000),
                rng.multivariate_normal(self.mean2, self.cov2, 2000),
            ))
            small_sample = rng.choice(full_sample, 200, replace=False)

            # Estimate the MI with two k values and sample sizes
            small_k1.append(estimate_mi(small_sample[:,0], small_sample[:,1], k=1))
            small_k3.append(estimate_mi(small_sample[:,0], small_sample[:,1], k=3))
            full_k2.append(estimate_mi(full_sample[:,0], full_sample[:,1], k=2))
            full_k40.append(estimate_mi(full_sample[:,0], full_sample[:,1], k=40))

        small_k1a = np.asarray(small_k1)
        small_k3a = np.asarray(small_k3)
        full_k2a = np.asarray(full_k2)
        full_k40a = np.asarray(full_k40)

        # With low sample size, increasing k should increase accuracy
        self.assertLess(np.mean(np.abs(small_k3a - expected)), np.mean(np.abs(small_k1a - expected)))
        self.assertAlmostEqual(np.median(small_k3a), expected, delta=0.02)

        # With high sample size, decreasing k should increase accuracy
        self.assertLess(np.mean(np.abs(full_k2a - expected)), np.mean(np.abs(full_k40a - expected)))
        self.assertAlmostEqual(np.median(full_k2a), expected, delta=0.01)


    def integrate_mi(self) -> float:
        def f(y: float, x: float) -> float:
            joint_dens = 0.5 * (mvnorm.pdf((x,y), self.mean1, self.cov1)\
                + mvnorm.pdf((x,y), self.mean2, self.cov2))
            x_dens = 0.5 * (norm.pdf(x, self.mean1[0], self.cov1[0,0])\
                + norm.pdf(x, self.mean2[0], self.cov2[0,0]))
            y_dens = 0.5 * (norm.pdf(y, self.mean1[1], self.cov1[1,1])\
                + norm.pdf(y, self.mean2[1], self.cov2[1,1]))

            return joint_dens * np.log(joint_dens / (x_dens * y_dens))
        
        # This rectangle contains most of the probability mass,
        # and we can safely reduce the accuracy too
        return dblquad(f, -6, 6, -6, 6, epsrel=1e-3)[0]
