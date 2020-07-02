# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

"""Benchmark our digamma versus SciPy digamma."""

import numpy as np
import timeit

setup = """
from ennemi._entropy_estimators import _psi as our_psi
from scipy.special import psi as scipy_psi
import numpy as np

data = np.arange(N)
"""

our_bench = "our_psi(data)"
scipy_bench = "scipy_psi(data)"

for (name, bench) in [ ("ennemi", our_bench), ("scipy", scipy_bench) ]:
    for n in [ 100, 400, 2000, 10000 ]:
        res = timeit.repeat(bench, setup, repeat=5, number=1, globals={"N": n})
        print(f"{name:<6}, N={n:<4}: min={np.min(res):<6.3} s, mean={np.mean(res):<6.3} s")
