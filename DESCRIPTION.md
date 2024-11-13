This package performs non-linear correlation analysis with mutual information (MI).
MI is an information-theoretical measure of dependency between two variables.
The package is designed for practical data analysis
with no theoretical background required.

**Features:**
- Non-linear correlation detection:
  - Mutual information between two variables, continous or discrete
  - Conditional MI with arbitrary-dimensional conditioning variables
  - Quick overview of many-variable datasets with pairwise MI estimation
- Practical data analysis:
  - Interfaces for evaluating multiple variable pairs and time lags with one call
  - Integrated with `pandas` data frames (optional)
  - Optimized and automatically parallelized estimation
  - Algorithms verified to work, so that you can focus on your data

This package depends only on NumPy and SciPy;
Pandas (2.x or newer) is suggested for more enjoyable data analysis.
Recent versions of NumPy 1.x and 2.x are supported.
Python 3.11+ on the latest macOS, Ubuntu and Windows versions
is officially supported.
Older `ennemi` versions have generally identical behavior if you need to run on older Python.

For more information on theoretical background and usage, please see the
[documentation](https://polsys.github.io/ennemi).
If you encounter any problems or have suggestions, please file an issue!

---

This package was initially developed at
[Institute for Atmospheric and Earth System Research (INAR)](https://www.helsinki.fi/en/inar-institute-for-atmospheric-and-earth-system-research),
University of Helsinki.
