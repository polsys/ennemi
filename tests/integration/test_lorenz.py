# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

"""Estimate the variable dependency in a coupled Lorenz system.

The set-up and reference results are described in Frenzel & Pompe (2007):
Partial Mutual Information for Coupling Analysis of Multivariate Time Series.
Physical Review Letters 99, doi:10.1103/PhysRevLett.99.204101

This test exercises especially the conditional MI code path, as the condition
is 7-dimensional. It also passes a two-dimensional cond_lag parameter.
"""

from ennemi import estimate_mi
import numpy as np
import unittest

class TestLorenz(unittest.TestCase):

    def simulate(self) -> np.ndarray:
        # Simulate the three Lorenz systems with 100x time resolution
        # Unlike Frenzel & Pompe, we use Euler's method for simplicity
        TIME_FACTOR = 100
        N = TIME_FACTOR * 1000
        BURNIN = 1000
        x = np.zeros((N + BURNIN, 3))
        y = np.zeros((N + BURNIN, 3))
        z = np.zeros((N + BURNIN, 3))

        TIME_STEP = 0.3
        SIGMA = 10
        R = 28
        B = 8/3
        K01 = 1
        K12 = 1
        TAU01 = 10 * TIME_FACTOR
        TAU12 = 15 * TIME_FACTOR

        # Set some initial values
        x[15*TIME_FACTOR,:] = 0.0
        y[15*TIME_FACTOR,:] = 1.0
        z[15*TIME_FACTOR,:] = 1.05

        for t in range(15*TIME_FACTOR, x.shape[0] - 1):
            # Calculate the derivatives
            xdot = SIGMA * (y[t] - x[t])
            ydot = R*x[t] - y[t] - x[t]*z[t]
            ydot[0] += K01 * y[t-TAU01,1]**2
            ydot[1] += K12 * y[t-TAU12,2]**2
            zdot = x[t]*y[t] - B*z[t]

            # Step the simulation
            x[t+1] = x[t] + (TIME_STEP/TIME_FACTOR)*xdot
            y[t+1] = y[t] + (TIME_STEP/TIME_FACTOR)*ydot
            z[t+1] = z[t] + (TIME_STEP/TIME_FACTOR)*zdot

        # Return every 100'th step from the three series' Y components
        return y[BURNIN::TIME_FACTOR]


    def verify_unconditional(self, data: np.ndarray) -> None:
        # Check the top row of Frenzel & Pompe, Figure 3
        lags = np.arange(-30, 30+1)

        # I(Y1; Y2) should have a peak at lag=10
        # However, the peak is very wide
        mi = estimate_mi(data[:,0], data[:,1], lags).flatten()
        argmax = np.argmax(mi)
        self.assertEqual(lags[argmax], 10)
        self.assertAlmostEqual(mi[argmax], 0.5, delta=0.05)
        self.assertGreaterEqual(np.sum(mi > 0.2), 4)

        # I(Y1; Y3) should have a peak at roughly lag=27
        mi = estimate_mi(data[:,0], data[:,2], lags).flatten()
        argmax = np.argmax(mi)
        self.assertIn(lags[argmax], [26, 27])
        self.assertAlmostEqual(mi[argmax], 0.24, delta=0.05)

        # I(Y2; Y3) should have a peak at lag=15
        mi = estimate_mi(data[:,1], data[:,2], lags).flatten()
        argmax = np.argmax(mi)
        self.assertEqual(lags[argmax], 15)
        self.assertAlmostEqual(mi[argmax], 0.6, delta=0.03)


    def verify_conditional(self, data: np.ndarray) -> None:
        # Check the bottom row of Figure 3
        # The conditioning variable is seven copies of the remaining variable
        # lagged symmetrically around the MI peak, as specified in the article.
        lags = np.arange(-30, 30+1)

        # I(Y1; Y2 | Y3) has a clear peak at lag=10, cond lags are 24,...,30
        # The MI value is larger than in the paper
        cond_lags = np.broadcast_to(np.arange(24, 30+1), (lags.shape[0], 7))
        cond = np.broadcast_to(np.column_stack((data[:,2],)), (data.shape[0], 7))
        cmi = estimate_mi(data[:,0], data[:,1], lags, cond=cond, cond_lag=cond_lags)
        argmax = np.argmax(cmi)
        self.assertEqual(lags[argmax], 10)
        self.assertEqual(np.sum(cmi > 0.1), 1)

        # I(Y1; Y3 | Y2) has no peak, cond lags are 7,...,13
        cond_lags = np.broadcast_to(np.arange(7, 13+1), (lags.shape[0], 7))
        cond = np.broadcast_to(np.column_stack((data[:,1],)), (data.shape[0], 7))
        cmi = estimate_mi(data[:,0], data[:,2], lags, cond=cond, cond_lag=cond_lags)
        self.assertTrue(np.all(cmi < 0.02))
        
        # I(Y2; Y3 | Y1) has a clear peak at lag=15, cond lags are 24,...,30
        cond_lags = np.broadcast_to(np.arange(-13, -7+1), (lags.shape[0], 7))
        cond = np.broadcast_to(np.column_stack((data[:,0],)), (data.shape[0], 7))
        cmi = estimate_mi(data[:,1], data[:,2], lags, cond=cond, cond_lag=cond_lags)
        argmax = np.argmax(cmi)
        self.assertEqual(lags[argmax], 15)
        self.assertAlmostEqual(cmi[argmax], 0.12, delta=0.05)


    def test_coupled_lorenz_systems(self) -> None:
        data = self.simulate()
        self.verify_unconditional(data)
        self.verify_conditional(data)
