---
title: Getting started
---

In this tutorial we will go through the various use cases of `ennemi`.
I assume that you know, at basic level, what mutual information means.
Before starting with your own projects, I strongly encourage you to still
read the [chapter on potential issues](potential-issues.md),
lest you want to discover those the hard way!

**Note:**
You may get slightly different results when running the examples.
The processor model, operating system and software versions
have subtle effects on floating point calculations.
The results should still be the same to several significant digits.


## Bivariate normal distribution

We begin by considering a two-variable normal distribution:
$$
(X, Y) \sim \mathrm{Normal}(\mu, \Sigma).
$$
We assume that both $X$ and $Y$ have unit variance.
The two variables may be correlated with the coefficient $\rho$.
In this case the covariance matrix becomes
$$
\Sigma = \begin{bmatrix}
1 & \rho\\
\rho & 1
\end{bmatrix}.
$$
There is an explicit formula for the mutual information between the two variables:
$$
\mathrm{MI}(X, Y) = -\frac 1 2 \ln \det \Sigma = -\frac 1 2 \ln (1-\rho^2).
$$

For example, if $\rho = 0.8$, we would expect $\mathrm{MI}(X, Y) \approx 0.51$.
We can test what `ennemi` outputs with the following code:
```python
from ennemi import estimate_mi
import numpy as np

rng = np.random.default_rng(1234)
rho = 0.8
cov = np.array([[1, rho], [rho, 1]])

data = rng.multivariate_normal([0, 0], cov, size=800)
x = data[:,0]
y = data[:,1]

print(estimate_mi(y, x))
```

The code prints
```
[[0.51039762]]
```
as would be expected.
The result is in double brackets because `estimate_mi()` always outputs a 2D array.
This will be discussed more below.



## Correlation coefficient
Mutual information may have any non-negative value.
For easier interpretation, the MI can be converted to correlation coefficient scale.
To continue the above example, we could execute
```python
print(estimate_mi(y, x, normalize=True))
```
to get the estimated correlation coefficient (`0.79980729`).

The returned coefficient approximately matches the **absolute value**
of the linear correlation coefficient after suitable transformations.
For example, consider the model $y = \sin(x) + \varepsilon$.
We calculate both the linear correlation and correlation from MI:
```python
from ennemi import estimate_mi
import numpy as np

rng = np.random.default_rng(1234)
x = rng.normal(0.0, 3.0, size=800)
y = np.sin(x) + rng.normal(0.0, 0.5, size=800)

print(f"From MI: {estimate_mi(y, x, normalize=True)[0,0]:.3}")
print(f"Pearson: {np.corrcoef(y, np.sin(x))[0,1]:.3}")
print(f"Pearson, untransformed: {np.corrcoef(y, x)[0,1]:.3}")
```

The two values are very close to each other.
Without the correct transformation, there is no linear correlation
between the two variables.
```
From MI: 0.824
Pearson: 0.812
Pearson, untransformed: 0.01993
```

There are some caveats to the above.
The two coefficients are theoretically equivalent only when the transformations
are monotonic.
Periodic transformations such as sine (as above) have additional requirements
on symmetry: `x` should be distributed evenly across periods.
Therefore, the returned coefficient **should be considered only approximate**.



## More variables

Let's extend the above example by adding another, unrelated variable.
The mutual information between independent variables is 0.

The `estimate_mi()` method accepts a 2D array for the `x` parameter.
In that case, it splits the array into columns $X_1$ to $X_m$,
and calculates $\mathrm{MI}(X_i, Y)$ for each $i = 1, \ldots, m$.

This interface is a shorter way of calculating the MI between several
$X$ variables and $Y$.
It is also much faster, because `ennemi` computes the estimates
in parallel whenever it is beneficial.

**Note:**
This should not be confused with _multivariate mutual information_,
a measure of interdependency between three or more variables together.
`ennemi` does not currently support multivariate MI,
as the results would be difficult to interpret.

Here's the above example updated with a new, independent variable $Z$:
```python
from ennemi import estimate_mi
import numpy as np

rng = np.random.default_rng(1234)
rho = 0.8
cov = np.array([[1, rho], [rho, 1]])

data = rng.multivariate_normal([0, 0], cov, size=800)
x = data[:,0]
y = data[:,1]
z = rng.normal(0, 1, size=800)

covariates = np.column_stack((x, z))

print(estimate_mi(y, covariates, normalize=True))
```

The code prints
```
[[ 0.79978795 -0.02110195]]
```
The first column gives the correlation between $(X, Y)$ and the second column between $(Z, Y)$.
As expected, the latter is very close to $0$.
Due to random uncertainty and properties of the estimation algorithm,
the result will not be exactly 0, and may even be negative.
(Negative values far from zero, including $-\infty$, are discussed in the
[chapter on potential issues](potential-issues.md).)

The `x` parameter is interpreted as `x[observation index, variable index]`.
This matches the order used by NumPy and Pandas libraries.
See below for an example of using Pandas for data import.



## Time lag

In many systems, variables are coupled with a time delay.
There may be a clear dependence between $Y(t)$ and $X(t-\Delta)$,
whereas the apparent link between $Y(t)$ and $X(t)$ may be weaker or nonexistent.

The time lag is specified by passing an integer or a list/array of integers as
the `lag` parameter to `estimate_mi()`.
The lags are applied to the $X_i$ variables and may be positive or negative
(in which case information flows from $Y$ to $X$ instead).

The $Y$ array is constrained so that it stays the same for all lags.
This is done in order to make the results comparable.
For example, if there are observations $Y(0), \ldots, Y(N-1)$
and the lags are $-1$, $0$ and $1$,
the array $Y(1), \ldots, Y(N-2)$ is compared with
$$
\begin{cases}
X(2), \ldots, X(N-1),\\
X(1), \ldots, X(N-2),\\
X(0), \ldots, X(N-3),
\end{cases}
$$
each in turn.

Let's consider a model where $Y(t) = X(t-1) + \varepsilon$,
and estimate the MI for various time lags:
```python
from ennemi import estimate_mi
import numpy as np

rng = np.random.default_rng(1234)
x = rng.gamma(1.0, 1.0, size=400)
y = np.zeros(400)
y[1:] = x[0:-1]
y += rng.normal(0, 0.01, size=400)

print(estimate_mi(y, x, lag=[1, 0, -1], normalize=True))
```

The code prints:
```
[[ 0.99975754]
 [-0.04579946]
 [-0.00918085]]
```
which means that $Y(t)$ depends strongly on $X(t-1)$, but not
at all on $X(t)$ or $X(t+1)$.
The rows of the result array correspond to the `lag` parameter.



## Combining the above with Pandas

In this example, we import the data set from a file using the Pandas package
and pass the imported data straight to `estimate_mi()`.
We calculate the MI for several time lags and plot the results with Matplotlib.

To try this example, download the [mi_example.csv](mi_example.csv) file.

```python
from ennemi import estimate_mi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("mi_example.csv")
time_lags = np.arange(-3, 6)

mi = estimate_mi(data["y"], data[["x1", "x2", "x3"]],
                 lag=time_lags, normalize=True)

plt.plot(time_lags, mi["x1"], label="$x_1$", marker="o")
plt.plot(time_lags, mi["x2"], label="$x_2$", marker="^")
plt.plot(time_lags, mi["x3"], label="$x_3$", marker="s")
plt.legend()
plt.xlabel("Time lag (steps)")
plt.ylabel("MI correlation")
plt.title("Mutual information between $y$ and...")
plt.show() # or plt.savefig(...)
```

The returned array is also a Pandas data frame, with
column names matching variable names and indices matching lag values:
```
          x1        x2        x3
-3 -0.017814 -0.005630 -0.006186
-2 -0.039053  0.391237  0.311316
-1 -0.039403 -0.003894 -0.037381
 0  0.280384 -0.028118  0.958259
 1  0.998803 -0.012764  0.349852
 2 -0.002487 -0.003956  0.150416
 3  0.137114  0.993299  0.275532
 4 -0.038925  0.368778 -0.041505
 5 -0.006512 -0.074845 -0.000150
```
![The mutual information for x1 peaks at lag 1, for x2 at lag 3 and for x3 stays near zero.](example_mi_plot.png)

From this plot we can deduce that:
- there is a connection between $Y$ and $X_1$, but only with lag 1,
- a slightly weaker connection between $Y$ and $X_2$, but only with lag 3,
- a connection between $Y$ and $X_3$ without any lag,
- and a weak connection with lag -2.

A word of warning: in a real time series, where the data is often autocorrelated,
the peaks will not be nearly as sharp.



## Conditional mutual information

Suppose that in our previous example, we know that there is a connection
between $X_1(t)$ and $X_2(t-2)$.

Now the question is: how much additional information does $X_1$ provide when
$X_2$ is already known?
We get this by calculating the conditional mutual information.
We pass the $X_2$ column as the `cond` parameter,
and specify that the condition should be lagged by additional two steps:

```python
from ennemi import estimate_mi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("mi_example.csv")
time_lags = np.arange(-3, 6)

mi = estimate_mi(data["y"], data[["x1", "x3"]], lag=time_lags,
                 cond=data["x2"], cond_lag=time_lags+2, normalize=True)
print(mi)

plt.plot(time_lags, mi["x1"], label="$x_1$", marker="o")
plt.plot([], []) # Keep the colors comparable
plt.plot(time_lags, mi["x3"], label="$x_3$", marker="s")
plt.legend()
plt.xlabel("Time lag (steps)")
plt.ylabel("Mutual information (nats)")
plt.title("Conditional mutual information between $y$ and...")
plt.show()
```

```
          x1        x3
-3 -0.034824  0.088609
-2 -0.086125 -0.015640
-1 -0.038222 -0.025096
 0 -0.096564  0.885572
 1  0.839362 -0.055681
 2 -0.067587 -0.014683
 3 -0.034984  0.101320
 4 -0.022871 -0.000989
 5 -0.047365 -0.045094
```
![The conditional MI for x1 peaks at lag 1, but x3 has even larger peak at lag 0.](example_cmi_plot.png)

Now we can see that $X_3$ is actually slightly more significant in determining $Y$ than $X_1$!
The dependence between $X_1$ and $X_2$ resulted in $X_1$ "borrowing"
some explanative power from $X_2$.

The example data was generated with
$$
\begin{align*}
X_2(t) &= \mathrm{Normal}(0.2, 4^2),\\
X_1(t) &= X_2(t-2) + \mathrm{Normal}(0, 0.2^2),\\
Y(t) &= X_1(t-1) + \mathrm{Normal}(0, 0.1^2),\\
X_3(t) &= Y(t) - X_2(t-3).
\end{align*}
$$
The reason for our result is that $X_3$ contains the full difference between
$Y$ and $X_2$ whereas there is still some random variation between $X_1$ and $Y$.

`ennemi` also supports multidimensional conditions.
This is useful when there are several common explanative variables.
To specify a multivariate condition, pass a two-dimensional array as the `cond`
parameter similarly to `x`.
Instead of comparing each variable separately as with `x`, the conditional
variables will be evaluated together.



## Masking observations

Sometimes, you want to use only a subset of the observations.
This is easy to do unless some variables are time lagged.
The time lags are easy to get the wrong way around, and
the calculation must be repeated for each lag value.

To make subsetting with time lags easier, `estimate_mi()` accepts a `mask` parameter.
The mask is an array (or list) of boolean values.
An `y` observation,
and the relevant `x` and `cond` observations after lagging,
is only used if the corresponding `mask` element is `True`.

Consider a model of two variables.
The first variable has a daily cycle.
The second variable follows the first with a time lag, but gets meaningful
values only in the daytime.

![A cosine curve, and another cosine curve truncated to be positive.](example_mask_plot.png)

To analyze the full data set, you would execute:
```python
from ennemi import estimate_mi
import matplotlib.pyplot as plt
import numpy as np

# One observation every 30 minutes
t = np.arange(4*48) / 2
rng = np.random.default_rng(1234)
x = -0.3 * np.cos(2 * np.pi * t / 24) + rng.normal(0, 0.01, len(t))
y = np.sqrt(np.maximum(0, -np.cos(2 * np.pi * (t-3) / 24)))\
                          + rng.normal(0, 0.01, len(t))

print(estimate_mi(y, x, lag=[0, 1, 2, 3], normalize=True))
```
The result is:
```
[[0.91572684]
 [0.92111013]
 [0.91883495]
 [0.93652557]]
```

To constrain to daytime observations of $Y$ only, replace the last line with
```python
mask = (t % 24 > 6) & (t % 24 < 18)
print(estimate_mi(y, x, lag=[0, 1, 2, 3], mask=mask, normalize=True))
```
This produces slightly different MI values:
```
[[0.84804286]
 [0.88699323]
 [0.90550901]
 [0.94374136]]
```
The three first correlations indicate that we actually had some information there:
if $X &lt; 0$, then $Y$ is probably zero.
The MI at the correct lag term increased a bit.

If some observations were missing altogether, indicated by a `NaN` value,
we could pass the `drop_nan` parameter to `estimate_mi()` and get the same results.
