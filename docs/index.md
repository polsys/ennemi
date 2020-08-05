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
  - Mutual information between two variables
  - Conditional MI with arbitrary-dimensional conditioning variables
  - Discrete-continuous MI
- Practical data analysis:
  - Interface for evaluating multiple variable pairs and time lags with one call
  - Integrated with `pandas` data frames (optional)
  - Optimized and automatically parallelized estimation

`ennemi` is currently developed by Petri Laarne ([@polsys](https://github.com/polsys)) at
[Institute for Atmospheric and Earth System Research (INAR)](https://www.helsinki.fi/en/inar-institute-for-atmospheric-and-earth-system-research),
University of Helsinki.
The package is published under the MIT License.


## Installation

This package is **beta** quality.
The algorithm is fairly stable
but breaking changes to the interface are still possible (if unlikely).

This package is available on PyPI:
```sh
pip install ennemi
```
(If your machine has multiple Python installations, you may need to run e.g. `pip3`.)


## Documentation topics

> **Note:** This documentation refers to the still-unreleased 1.0.0beta1 version.
> There are some minor API changes, as seen in the
> [draft release notes](https://gist.github.com/polsys/f0757723b73194a0bccb9043e7c75e47).
> We aim to publish the version in early August.

- **[What is entropy?](what-is-entropy.md)**
  A quick overview of the theory and terminology.
- **[Getting started](tutorial.md).**
  This tutorial covers the basic use cases through examples.
- **[Case study](kaisaniemi.md).**
  An example workflow with real data.
- **[Potential issues](potential-issues.md).**
  Common issues and how to solve them.
- **[API reference](api-reference.md).**
  All methods and their parameters.
- **[Code snippets](snippets.md).**
  Example utilities that may be useful.