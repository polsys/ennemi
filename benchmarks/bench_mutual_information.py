# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

"""Benchmarks for ordinary and conditional MI estimation."""

import numpy as np
import timeit

setup = """
from ennemi import estimate_mi
import numpy as np

rng = np.random.default_rng(0)
t = np.arange(N)

a = np.sin(t)
b = np.cos(t) + rng.normal(0, 1, size=N)
c = rng.normal(2, 5, size=N)
d = rng.gamma(1.0, 1.0, size=N)
e = b + 2.0*c
f = c + 2.0*d
s = np.sign(d)

data = np.asarray([a, b, c]).T
cond2 = np.asarray([e, f]).T
"""

mi_bench = "estimate_mi(d, data, lag=[-1, 0, 1, 2], k=3)"
cmi_bench = "estimate_mi(d, data, lag=[-1, 0, 1, 2], k=3, cond=e)"
cmi2_bench = "estimate_mi(d, data, lag=[-1, 0, 1, 2], k=3, cond=cond2)"
dmi_bench = "estimate_mi(s, data, lag=[-1, 0, 1, 2], k=3, discrete_y=True)"
dcmi_bench = "estimate_mi(s, data, lag=[-1, 0, 1, 2], k=3, cond=e, discrete_y=True)"

for (name, bench) in [ ("MI", mi_bench),
                       ("CMI", cmi_bench),
                       ("CMI2", cmi2_bench),
                       ("DMI", dmi_bench),
                       ("DCMI", dcmi_bench) ]:
    for n in [ 100, 400, 1600, 6400 ]:
        res = timeit.repeat(bench, setup, repeat=5, number=1, globals={"N": n})
        print(f"{name:<4}, N={n:<4}: min={np.min(res):<6.3} s, mean={np.mean(res):<6.3} s")
