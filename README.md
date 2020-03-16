# ennemi
_Easy-to-use Nearest Neighbor Estimation of Mutual Information._

This Python 3 package provides simple methods for estimating mutual information of continuous variables.
With one method call, you can estimate several variable pairs and time lags at once.

The package uses the nearest neighbor method as described by Kraskov et al. in
_Estimating Mutual Information_, Physical Review E **69**:6 (2004),
[doi:10.1103/PhysRevE.69.066138](https://dx.doi.org/10.1103/PhysRevE.69.066138).
The variant for conditional mutual information is described in Frenzel & Pompe,
_Partial Mutual Information for Coupling Analysis of Multivariate Time Series_,
Physical Review Letters **99**:20 (2007),
[doi:10.1103/PhysRevLett.99.204101](https://dx.doi.org/10.1103/PhysRevLett.99.204101).

**NOTE:** This package is **alpha** quality. Likelihood of breaking changes to the API is high!


## Getting started

This package requires Python 3.6 or higher.
The only dependency is NumPy.

In the future, the package will be published on PyPI.
For now, see the building instructions.

For documentation, please see https://polsys.github.io/ennemi.


## Building

The tests depend on SciPy, so you need that installed in addition to NumPy.

To install the package in development mode, clone this repository and execute
```sh
pip install -e .
```
in the repository root folder.
(If your machine has multiple Python installations, you may need to run e.g. `pip3`.)


## Citing

This work is licensed under the MIT license.
In short, you are allowed to use, modify and distribute it as you wish, as long as
the original copyright notice is preserved.
This software is provided "as is", functioning to the extent of passing unit tests in the `tests` directory.

Once the project has matured a bit more, a DOI will be requested via Zenodo.
