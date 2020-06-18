# ennemi
_Easy-to-use Nearest Neighbor Estimation of Mutual Information_

[![Continuous Integration](https://github.com/polsys/ennemi/workflows/Continuous%20Integration/badge.svg?event=push)](https://github.com/polsys/ennemi/actions)
[![Integration Tests](https://github.com/polsys/ennemi/workflows/Integration%20Tests/badge.svg)](https://github.com/polsys/ennemi/actions)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=polsys_ennemi&metric=coverage)](https://sonarcloud.io/dashboard?id=polsys_ennemi)
[![DOI](https://zenodo.org/badge/247088713.svg)](https://zenodo.org/badge/latestdoi/247088713)

This Python 3 package provides simple methods for estimating mutual information of continuous variables.
With one method call, you can estimate several variable pairs and time lags at once.
Both unconditional and conditional MI (including multivariate condition) are supported.

The package uses the nearest neighbor method as described by Kraskov et al. in
_Estimating Mutual Information_, Physical Review E **69**:6 (2004),
[doi:10.1103/PhysRevE.69.066138](https://dx.doi.org/10.1103/PhysRevE.69.066138).
The variant for conditional mutual information is described in Frenzel & Pompe,
_Partial Mutual Information for Coupling Analysis of Multivariate Time Series_,
Physical Review Letters **99**:20 (2007),
[doi:10.1103/PhysRevLett.99.204101](https://dx.doi.org/10.1103/PhysRevLett.99.204101).

**NOTE:** This package is **alpha** quality.
Even though the algorithm is stable, there may still be breaking changes to the API.
You can see the roadmap of planned additions on the
[Milestones page](https://github.com/polsys/ennemi/milestones).


## Getting started

This package requires Python 3.6 or higher,
and it is tested to work on the latest versions of Ubuntu, macOS and Windows.
The only dependency is NumPy.

This package is available on PyPI:
```sh
pip install ennemi
```
(If your machine has multiple Python installations, you may need to run e.g. `pip3`.)

For documentation, please see https://polsys.github.io/ennemi.


## Building

The tests depend on SciPy and pandas, so you need those installed in addition to NumPy.
Additionally, `pytest` and `mypy` are required for building the project.
All of these are installed by the "extras" syntax of `pip`.

To install the package in development mode, clone this repository and execute
```sh
pip install -e .[dev]
```
in the repository root folder.

All methods, including tests, are type annotated and checked with `mypy`.
The CI script runs the check automatically on each pushed commit.
To run the check yourself, execute
```sh
python -m mypy --disallow-untyped-defs ennemi/ tests/unit tests/integration
```

Please see also the [contribution guidelines](CONTRIBUTING.md).


## Citing

This work is licensed under the MIT license.
In short, you are allowed to use, modify and distribute it as you wish, as long as
the original copyright notice is preserved.
This software is provided "as is", functioning to the extent of passing
the unit and integration tests in the `tests` directory.

This package is archived on Zenodo.
The DOI [10.5281/zenodo.3834018](https://doi.org/10.5281/zenodo.3834018)
always resolves to the latest version of `ennemi`.
For reproducibility, you should cite the exact version of the package you have used.
To do so, use the DOI given on the [Releases](https://github.com/polsys/ennemi/releases) page or on Zenodo.

In the future, a journal article may become the canonical reference for this package.
You should still also cite the package version you have used.

This package is maintained by Petri Laarne at
[Institute for Atmospheric and Earth System Research (INAR)](https://www.helsinki.fi/en/inar-institute-for-atmospheric-and-earth-system-research),
University of Helsinki.
Please feel free to contact me about any questions or suggestions related to this project.
