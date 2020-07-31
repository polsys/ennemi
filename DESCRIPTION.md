This package performs non-linear correlation analysis with mutual information (MI).
MI is an information-theoretical measure of dependency between two variables.
The package is designed for practical data analysis
with no theoretical background required.

**Features:**
- Non-linear correlation detection:
  - Mutual information between two variables
  - Conditional MI with arbitrary-dimensional conditioning variables
  - Discrete-continuous MI
- Practical data analysis:
  - Interface for evaluating multiple variable pairs and time lags with one call
  - Integrated with `pandas` data frames (optional)
  - Optimized and automatically parallelized estimation

This package depends only on NumPy and SciPy;
Pandas is suggested for more enjoyable data analysis.
Python 3.6+ on the latest macOS, Ubuntu and Windows versions
are officially supported.

This project is still in **beta** status: breaking changes are unlikely but possible.
For more information on theoretical background and usage, please see the
[documentation](https://polsys.github.io/ennemi).
If you encounter any problems or have suggestions, please file an issue!

---

This package has been developed at
[Institute for Atmospheric and Earth System Research (INAR)](https://www.helsinki.fi/en/inar-institute-for-atmospheric-and-earth-system-research),
University of Helsinki.
