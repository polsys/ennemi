---
title: Code snippets
---

This is a collection of random utility functions
that are out of scope for the package itself.
These code snippets are placed in the public domain;
you may use and modify them without attribution.

# Progress reporting

These functions create a callback that prints a line for every 10% of estimation tasks done.
They calculate the number of tasks by the `x` and `lag` array shape.
The optional `factor` parameter can be used if you average several estimates
together; the same callback can be passed to each estimation call.

```python
def get_estimate_mi_callback(x, lags, factor=1):
    """Return a callback that prints a line for every 10% of estimation."""
    total_tasks = x.shape[1] * len(lags) * factor
    tasks_done = 0

    def callback(var_index, lag):
        nonlocal tasks_done
        tasks_done += 1

        cur_percent = (tasks_done / total_tasks) * 100
        prev_percent = ((tasks_done - 1) / total_tasks) * 100

        if cur_percent // 10 > prev_percent // 10:
            print(f"{datetime.now():%X}: {int(cur_percent):>3}% ({tasks_done}/{total_tasks})")

    return callback

def get_pairwise_mi_callback(x, factor=1):
    """Return a callback that prints a line for every 10% of estimation."""
    total_tasks = (x.shape[1] * (x.shape[1] - 1) // 2) * factor
    tasks_done = 0

    def callback(var_index, lag):
        nonlocal tasks_done
        tasks_done += 1

        cur_percent = (tasks_done / total_tasks) * 100
        prev_percent = ((tasks_done - 1) / total_tasks) * 100

        if cur_percent // 10 > prev_percent // 10:
            print(f"{datetime.now():%X}: {int(cur_percent):>3}% ({tasks_done}/{total_tasks})")

    return callback
```

To use the callback, pass the returned value to `estimate_mi`/`pairwise_mi`
(or their `..._corr` counterparts)
as the optional `callback` keyword parameter:
```python
y, covariates = ...
lags = [0, 1, 2, 3]
callback = get_estimate_mi_callback(covariates, lags)

mi = estimate_mi(y, covariates, lags, callback=callback)
```
