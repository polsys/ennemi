# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

"""Plot a visualization of the MI estimation performance.

Requires Matplotlib (not installed by 'pip install ennemi[dev]').
"""

import matplotlib.pyplot as plt
import numpy as np
import timeit

setup = """
from ennemi import estimate_mi
import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(size=n)
y = rng.normal(size=n)
z = rng.normal(size=(n, 2))
"""

all_ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
          15, 20, 25, 30, 35, 40, 45, 50]
all_ns = np.linspace(200, 25000, 20, dtype=np.int32)

names = ["2D condition", "1D condition", "Unconditional"]
benches = [
    "estimate_mi(y,x,k=k,cond=z)",
    "estimate_mi(y,x,k=k,cond=z[:,0])",
    "estimate_mi(y,x,k=k)",
]

# The N plot
print("Using different values of n...")

results_n = np.empty((len(all_ns), len(benches)))
for (i, n) in enumerate(all_ns):
    for (j, test) in enumerate(benches):
        print(f"n={n:>4}, {names[j]}")
        times = timeit.repeat(test, setup,
                              repeat=10, number=1,
                              globals={"k":5, "n":n})
        best = np.min(times)
        results_n[i,j] = best

# The k plot
print("Using different values of k...")

results_k = np.empty((len(all_ks), len(benches)))
for (i, k) in enumerate(all_ks):
    for (j, test) in enumerate(benches):
        print(f"k={k:>2}, {names[j]}")
        times = timeit.repeat(test, setup,
                              repeat=10, number=1,
                              globals={"k":k, "n":10000})
        best = np.min(times)
        results_k[i,j] = best

# Draw the plots
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4.5), constrained_layout=True, sharey=True)

for (i, name) in enumerate(names):
    ax1.plot(all_ns, results_n[:,i], label=name, marker=".")
    ax2.plot(all_ks, results_k[:,i], label=name, marker=".")

ax1.set_xlabel("$n$ (fixed $k = 5$)")
ax1.set_ylabel("Execution time (s)")
ax1.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
ax1.set_xticks([0, 2500, 7500, 12500, 17500, 22500], minor=True)
ax1.set_yticks([0.25, 0.75, 1.25, 1.75, 2.25], minor=True)
ax1.grid()
ax1.legend()

ax2.set_xlabel("$k$ (fixed $n = 10000$)")
ax2.grid()
ax2.legend()

plt.savefig("mi_perf.pdf", transparent=True)
