This package implements the estimation of mutual information (MI) between
two continuous variables using a nearest-neighbor algorithm
([Kraskov et al. 2004](https://dx.doi.org/10.1103/PhysRevE.69.066138)).
Mutual information is an information-theoretical measure of
dependency between two variables.

The interface is minimal and aimed at practical data analysis:

- Support for masking and time lags between variables
- Conditional MI with arbitrary-dimensional conditions
- Optimized algorithm and parallel processing of multiple estimation tasks
- Normalization of MI to correlation coefficient scale
- Optional integration with `pandas` data frame types (no install-time dependency)

This package depends only on NumPy.
Support for Python 3.6+ on the latest macOS, Ubuntu and Windows versions
is officially tested.

This project is still in **alpha** status and interface changes are possible.
For more information on theoretical background and usage, please see the
[documentation](https://polsys.github.io/ennemi).
If you encounter any problems or have suggestions, please file an issue!
