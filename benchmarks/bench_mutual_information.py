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

data = np.asarray([a, b, c])
"""

mi_bench = "estimate_mi(d, data, lag=[-1, 0, 1, 2], k=3)"
cmi_bench = "estimate_mi(d, data, lag=[-1, 0, 1, 2], k=3, cond=e)"

for (name, bench) in [ ("MI", mi_bench), ("CMI", cmi_bench) ]:
    for n in [ 100, 400, 1600 ]:
        res = timeit.repeat(bench, setup, repeat=5, number=1, globals={"N": n})
        print(f"{name:<3}, N={n:<4}: min={np.min(res):<6.3} s, mean={np.mean(res):<6.3} s")
