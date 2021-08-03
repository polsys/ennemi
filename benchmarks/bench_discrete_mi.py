# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

"""Benchmarks for discrete MI (both basic and conditional) estimation."""

import numpy as np
import timeit

setup = """
from ennemi import estimate_mi
import numpy as np

rng = np.random.default_rng(0)

a = rng.choice(["a", "b", "c", "d"], N, p=[0.2, 0.4, 0.3, 0.1])
b = rng.choice(["a", "b", "c", "d"], N)
c = rng.choice(np.arange(20), N)
d = rng.choice(np.arange(10), N)
e = c + 2*d
s = (a == b)

data = np.asarray([a, b, c]).T
cond2 = np.asarray([b, e]).T
"""

uncond_bench = "estimate_mi(a, data, lag=[-1, 0, 1, 2], discrete_y=True, discrete_x=True)"
cond_bench = "estimate_mi(a, data, lag=[-1, 0, 1, 2], discrete_y=True, discrete_x=True, cond=b)"
cond2_bench = "estimate_mi(a, data, lag=[-1, 0, 1, 2], discrete_y=True, discrete_x=True, cond=cond2)"

for (name, bench) in [ ("discrete MI", uncond_bench),
                       ("discrete CMI", cond_bench),
                       ("discrete CMI2", cond2_bench) ]:
    for n in [ 400, 1600, 6400, 25600 ]:
        res = timeit.repeat(bench, setup, repeat=5, number=1, globals={"N": n})
        print(f"{name:<13}, N={n:<4}: min={np.min(res):<6.3} s, mean={np.mean(res):<6.3} s")
