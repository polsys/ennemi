---
title: Potential issues
---

There are a few things you should know when using `ennemi` in your projects.
In case you encounter an issue not mentioned here, please get in touch!



## Discrete/duplicate observations
If `estimate_mi()` returns a negative number near zero (like -0.03),
there is no issue.
Mathematically, mutual information is always non-negative,
but the estimation algorithm is not exact.
However, if the result is further from zero, or even `-inf`,
that certainly is an issue.

The estimation algorithm assumes that the variables are
sampled from a continuous distribution.
This assumption can be violated in two ways:
- The distribution is discrete.
  The data is either discrete or is recorded at low resolution,
  causing duplicate data points.
- The distribution is neither discrete nor continuous.
  Censored or truncated distributions fall into this category.

As of this writing, `ennemi` does not yet support discrete distributions.
For discrete-discrete MI, other packages are available.
The discrete-continuous case may be implemented in the future, as
[Ross (2014)](https://dx.plos.org/10.1371/journal.pone.0087357)
has derived a suitable version of the algorithm.

For low-resolution or censored data, the suggestion of
[Kraskov et al. (2004)](https://link.aps.org/doi/10.1103/PhysRevE.69.066138)
is to add some low-amplitude noise to the non-continuous variable.
As an example, here is a censored version of the bivariate Gaussian example:
```python
from ennemi import estimate_mi
import numpy as np

rng = np.random.default_rng(1234)
rho = 0.8
cov = np.array([[1, rho], [rho, 1]])

data = rng.multivariate_normal([0.5, 0.5], cov, size=800)
x = np.maximum(0, data[:,0])
y = np.maximum(0, data[:,1])

print("MI:", estimate_mi(y, x))
print("On one or both axes:", np.mean((x == 0) | (y == 0)))
print("At origin:", np.mean((x == 0) & (y == 0)))
```
This code prints
```
MI: [[-0.49151162]]
On one or both axes: 0.4025
At origin: 0.22
```
As many of the observations lie on the x or y axis, and most of those at $(0, 0)$,
the algorithm produces a clearly incorrect result.
The fix is to add
```python
x += rng.normal(0, 1e-6, size=800)
y += rng.normal(0, 1e-6, size=800)
```
before the call to `estimate_mi()`.
With this fix, the code now prints
```
MI: [[0.41807897]]
```
a better approximation of the true value.
This is still an approximation, as the true distribution is non-continuous,
but in many cases the value is close to the theoretical result.
(The computation involves the more general theory of Lebesgue integral.)



## Skewed or high-variance distributions
Mutual information is invariant under strictly monotonic transformations.
Such transformations include
- For all variables: addition with a constant,
  multiplication with a non-zero constant and exponentiation;
- For positive variables: logarithm, powers and roots.

However, with sample sizes less than $\infty$,
the estimation algorithm may produce different results
between original and transformed variables.
For this reason it is recommended to scale the variables to have
roughly unit variance, symmetric distributions.
As said, this transformation will not change the actual MI,
but will improve the accuracy of the estimate.

For example, here is the bivariate Gaussian example modified to be
lognormal in both marginal directions,
with greater variance in the `x` variable.
As mentioned in the tutorial, $\mathrm{MI}(X, Y) \approx 0.51$ and
the exponentiation does not change this.
```python
from ennemi import estimate_mi
import numpy as np

rng = np.random.default_rng(1234)
rho = 0.8
cov = np.array([[1, rho], [rho, 1]])

data = rng.multivariate_normal([0, 0], cov, size=800)
x = np.exp(5 * data[:,0])
y = np.exp(data[:,1])

print(estimate_mi(y, x))
```

Running the code outputs
```
[[0.28945921]]
```

This demonstrates that you should normalize all variables before running
the estimation.



## Autocorrelation
The estimation method requires that the observations are
independent and identically distributed.
If the samples have significant autocorrelation, the first assumption does not hold.
In this case, the algorithm may return too high MI values.

In this example, each point is present three times:
the additional occurrences have some added random noise.
This simulates the autocorrelation between the samples.
```python
from ennemi import estimate_mi
import numpy as np

rng = np.random.default_rng(1234)
rho = 0.8
cov = np.array([[1, rho], [rho, 1]])

data = rng.multivariate_normal([0, 0], cov, size=800)
x = np.concatenate((data[:,0],
    data[:,0] + rng.normal(0, 0.01, size=800),
    data[:,0] + rng.normal(0, 0.01, size=800)))
y = np.concatenate((data[:,1],
    data[:,1] + rng.normal(0, 0.01, size=800),
    data[:,1] + rng.normal(0, 0.01, size=800)))

print(estimate_mi(y, x))
```

Running the code outputs
```
[[1.02612906]]
```
a significantly higher value than the $\approx 0.51$ expected of
non-autocorrelated samples.

The reason for this error is that the approximated density is peakier than
it should be.
Each point has effectively three times the density it should have,
while the gaps between points are unaffected.
We can see this in a Gaussian kernel density estimate of the sample.
For simplicity, we plot only the marginal `y` density.
```python
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

baseline_kernel = gaussian_kde(data[:,1])
autocorrelated_kernel = gaussian_kde(y)

t = np.linspace(-2, 2, 100)
plt.plot(t, baseline_kernel(t), "--", label="Baseline")
plt.plot(t, autocorrelated_kernel(t), label="Autocorrelated")
plt.legend()
plt.show()
```

![The autocorrelated density has more bumps than the roughly Gaussian baseline.](autocorrelation_kde.png)

The way of fixing this depends on your data.
Does it make sense to look at deseasonalized or differenced data?
Can you reduce the sampling frequency so that the autocorrelation is smaller?



## Improving performance
Even though `ennemi` uses good algorithms (in terms of asymptotic performance),
it is not designed for high-performance computing.
The library is implemented in Python (for easy portability)
and can use efficient NumPy methods only to some extent.
If you need absolutely highest performance, you should look at alternative
libraries or implement your own in a compiled, vectorized language.

That being said, `ennemi` is fairly fast for reasonable problem sizes,
and here are some tips to get the most out of your resources.


### Basics
- Close other apps if possible.
- Desktop computers are usually faster than laptops,
  especially in parallel tasks (see below).
- If you use a laptop, plug it in.
  The performance is usually limited when running on battery power.
- The golden rule of performance optimization: always measure how fast it is!


### Parallelize the calculation
If you call `estimate_mi()` with several variables and/or time lags,
the calculation will be split into as many concurrent tasks
as you have processor cores.
With a four-core processor, you may expect the run time to be cut in half.
(There is some overhead from starting new Python processes and copying the data.)

This means that instead of writing
```python
# This code is not optimal
for lag in all_the_lags:
    for var in variables:
        estimate_mi(y, data[var], lag)
```
you should write
```python
estimate_mi(y, data, all_the_lags)
```
