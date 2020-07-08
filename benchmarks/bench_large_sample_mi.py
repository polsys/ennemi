# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

"""Benchmark for a large sample MI between two variables only.

As opposed to bench_mutual_information.py, this benchmark cannot be improved
by parallelization. This tests only the quality of the underlying algorithm.
"""

import numpy as np
import timeit

setup = """
from ennemi import estimate_mi
import numpy as np

rng = np.random.default_rng(0)
cov = np.array([[1, 0.8], [0.8, 1]])
data = rng.multivariate_normal([0, 0], cov, size=20000)

x = data[:,0]
y = data[:,1]
"""

# Warm up so that possible JIT compilation does not show up in results
print("Warming up...")
warmup = timeit.repeat("estimate_mi(y[:50], x[:50], k=3)", setup, repeat=1, number=1)
print(f"Warm-up: {np.min(warmup):.3} s")
print()

bench = "estimate_mi(y, x, k=8)"
res = timeit.repeat(bench, setup, repeat=5, number=1)
print(f"N=20000, k=8: min={np.min(res):<6.3} s, mean={np.mean(res):<6.3} s")
