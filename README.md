# ennemi
_(Easy Nearest Neighbor Estimation of Mutual Information)_

[![Continuous Integration](https://github.com/polsys/ennemi/workflows/Continuous%20Integration/badge.svg?event=push)](https://github.com/polsys/ennemi/actions)
[![Integration Tests](https://github.com/polsys/ennemi/workflows/Integration%20Tests/badge.svg)](https://github.com/polsys/ennemi/actions)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=polsys_ennemi&metric=coverage)](https://sonarcloud.io/dashboard?id=polsys_ennemi)
[![DOI](https://zenodo.org/badge/247088713.svg)](https://zenodo.org/badge/latestdoi/247088713)

This Python 3 package provides simple methods for estimating mutual information of continuous and discrete variables.
With one method call, you can estimate several variable pairs and time lags at once.
Both unconditional and conditional MI (including multivariate condition) are supported.
The interface is aimed at practical, non-linear correlation analysis.

The package uses the nearest neighbor methods decscribed in:
- Kraskov et al. (2004): Estimating mutual information. Physical Review E 69.
  [doi:10.1103/PhysRevE.69.066138](https://dx.doi.org/10.1103/PhysRevE.69.066138).
- Frenzel & Pompe (2007): Partial Mutual Information for Coupling Analysis of
  Multivariate Time Series. Physical Review Letters 99.
  [doi:10.1103/PhysRevLett.99.204101](https://dx.doi.org/10.1103/PhysRevLett.99.204101).
- Ross (2014): Mutual Information between Discrete and Continuous Data Sets.
  PLoS ONE 9.
  [doi:10.1371/journal.pone.0087357](https://dx.doi.org/10.1371/journal.pone.0087357).

The latest source code on GitHub might be less stable than the released version on PyPI.
You can see the roadmap of planned additions on the
[Milestones page](https://github.com/polsys/ennemi/milestones).
See also the [support statement](docs/support.md).
You can also follow the development by clicking `Watch releases` on the GitHub page.


## Getting started

This package requires Python 3.8 or higher,
and it is tested to work on the latest versions of Ubuntu, macOS and Windows.
The only hard dependencies are reasonably recent versions of NumPy and SciPy;
Pandas is strongly suggested for more enjoyable data analysis.

This package is available on PyPI:
```sh
pip install ennemi
```
(If your machine has multiple Python installations, you may need to run e.g. `pip3`.)

For documentation, please see https://polsys.github.io/ennemi.


## Building

The tests depend on pandas, so you need that installed in addition.
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
python -m mypy ennemi/ tests/unit tests/integration tests/pandas
```
in the repository root (configuration is stored in `mypy.ini` file).

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

If you want to cite an article
(although you should still mention the version number you used), the reference is:
```
@article{ennemi,
  title = {ennemi: Non-linear correlation detection with mutual information},
  journal = {SoftwareX},
  volume = {14},
  pages = {100686},
  year = {2021},
  doi = {https://doi.org/10.1016/j.softx.2021.100686},
  author = {Petri Laarne and Martha A. Zaidan and Tuomo Nieminen}
}
```

This package is maintained by Petri Laarne, and was initially developed at
[Institute for Atmospheric and Earth System Research (INAR)](https://www.helsinki.fi/en/inar-institute-for-atmospheric-and-earth-system-research),
University of Helsinki.
Please feel free to contact me at `firstname.lastname@helsinki.fi`
about any questions or suggestions related to this project.
