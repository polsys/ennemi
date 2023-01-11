---
title: Documentation
---

## In brief

`ennemi` is a Python 3 package for estimating mutual information and other
information-theoretic measures of continuous variables.
These measures can be used for non-linear correlation analysis.
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

`ennemi` is maintained by Petri Laarne ([@polsys](https://github.com/polsys)) and was initially developed at
[Institute for Atmospheric and Earth System Research (INAR)](https://www.helsinki.fi/en/inar-institute-for-atmospheric-and-earth-system-research),
University of Helsinki.
The package is published under the MIT License.


## Documentation topics

- **[What is entropy?](what-is-entropy.md)**
  A quick overview of the theory and terminology.
- Tutorials:
  - **[Getting started](tutorial.md).**
    This tutorial covers the basic use cases through examples.
  - **[Case study](kaisaniemi.md).**
    An example workflow with real data.
  - **[Working with discrete data](discrete-data.md).**
    Estimation and interpretation when some variables are discrete.
- **[Potential issues](potential-issues.md).**
  Common issues and how to solve them.
- **[API reference](api-reference.md).**
  All methods and their parameters.
- **[Code snippets](snippets.md).**
  Example utilities that may be useful.
- **[Future support](support.md).**
  What to expect from the maintenance.


## Installation

This package is available on PyPI:
```sh
pip install ennemi
```


## Citing

If you use this software in scientific work,
please consider mentioning the exact version number for reproducibility.
Versions are citable and archived on Zenodo
([doi:10.5281/zenodo.3834018](https://doi.org/10.5281/zenodo.3834018)).

This package is described in the following articles:

- Code: Laarne, Zaidan, Nieminen: _ennemi: Non-linear correlation detection with mutual information_,
  SoftwareX, 2021.
  [doi:10.1016/j.softx.2021.100686](https://dx.doi.org/10.1016/j.softx.2021.100686).
- Case study (with downloadable source code for all examples):
  Laarne, Amnell, Zaidan, Mikkonen, Nieminen:
  _Exploring Non-Linear Dependencies in Atmospheric Data with Mutual Information_,
  Atmosphere, 2022.
  [doi:10.3390/atmos13071046](https://dx.doi.org/10.3390/atmos13071046).

Several research groups have already published papers that cite `ennemi`.
We hope that you find the package useful for your work!
