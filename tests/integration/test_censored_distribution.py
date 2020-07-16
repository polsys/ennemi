# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

"""A Gaussian distribution censored on one axis.

There is no analytical expression for the MI, and the algorithm in ennemi
expects a continuous distribution. Nevertheless, ennemi seems to work fine.
"""

from ennemi import estimate_mi
from scipy.integrate import quad, dblquad
from scipy.stats import norm, multivariate_normal as mvnorm
import numpy as np
import unittest

class TestCensoredDistribution(unittest.TestCase):
    def setUp(self) -> None:
        # A normal distribution offset to the right, and censored at x < 0
        self.cov = np.asarray([[1, 0.6], [0.6, 1]])
        self.mean = np.asarray([1, 0])


    def test_mi(self) -> None:
        # Estimate the actual MI by numerical integration
        expected = self.integrate_mi()

        # Sample from the distribution
        # The automatic preprocessing adds minor noise to the data
        rng = np.random.default_rng(0)
        sample = rng.multivariate_normal(self.mean, self.cov, 4000)
        sample[:,0] = np.maximum(0, sample[:,0])

        # The estimated MI should be very close to the numerical one
        actual = estimate_mi(sample[:,0], sample[:,1])

        self.assertAlmostEqual(actual, expected, delta=0.015)


    def integrate_mi(self) -> float:
        # We divide the integral into three parts:
        #  - x < 0, y arbitrary: identically zero
        #  - x = 0, y arbitrary: a discontinuous probability mass
        #  - x > 0, y arbitrary: the ordinary normal density
        #
        # It is more than sufficient to consider only (-4, 6) x (-6, 6)
        # when computing the integrals.

        # The density along x = 0
        def point_func(y: float) -> float:
            # The joint density is P(x <= 0, y)
            joint_dens = quad(lambda x: mvnorm.pdf((x,y), self.mean, self.cov), -4, 0, epsrel=5e-3)[0]

            # The marginal x density comes from the cumulative density
            x_dens = norm.cdf(0, self.mean[0], self.cov[0,0])
            y_dens = norm.pdf(y, self.mean[1], self.cov[1,1])

            return joint_dens * np.log(joint_dens / (x_dens * y_dens))
        
        # The density on x > 0
        def cont_func(y: float, x: float) -> float:
            # Ordinary joint and marginal densities
            joint_dens = mvnorm.pdf((x,y), self.mean, self.cov)
            x_dens = norm.pdf(x, self.mean[0], self.cov[0,0])
            y_dens = norm.pdf(y, self.mean[1], self.cov[1,1])

            return joint_dens * np.log(joint_dens / (x_dens * y_dens))
        
        point_part = quad(point_func, -6, 6)
        cont_part = dblquad(cont_func, 0, 6, -6, 6, epsrel=1e-3)

        return point_part[0] + cont_part[0]
