# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

#
# STEP 1: Import the data
#
from ennemi import estimate_corr, pairwise_corr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The first column contains a datetime index
data = pd.read_csv("kaisaniemi.csv", index_col=0, parse_dates=True)
print(data.head())


#
# STEP 2: Preprocess
#
# Nothing to be done, because the distributions are roughly symmetric


#
# STEP 3: Create a mask
#
afternoon_mask = (data.index.hour == 13)


#
# STEP 4: Plot pairwise MI
#
pairwise = pairwise_corr(data, mask=afternoon_mask)

# Plot a matrix where the color represents the correlation coefficient.
# We clip the color values at 0.2 because of significant random noise,
# and at 0.8 to make the color constrast larger.
fig, ax = plt.subplots(figsize=(8,6))
mesh = ax.pcolormesh(pairwise, vmin=0.2, vmax=0.8)
fig.colorbar(mesh, label="MI correlation coefficient", extend="both")

# Show the variable names on the axes
ax.set_xticks(np.arange(len(data.columns)) + 0.5)
ax.set_yticks(np.arange(len(data.columns)) + 0.5)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
ax.set_title("Unconditional MI at 15:00 local time")

plt.savefig("casestudy_pairwise.png", transparent=True)


#
# STEP 5: The same but conditional on DOY
#

# Now we pass the 'cond' parameter
pairwise_doy = pairwise_corr(data, mask=afternoon_mask,
    cond=data["DayOfYear"])

# The same plotting code as above
fig, ax = plt.subplots(figsize=(8,6))
mesh = ax.pcolormesh(pairwise_doy, vmin=0.2, vmax=0.8)
fig.colorbar(mesh, label="MI correlation coefficient", extend="both")

ax.set_xticks(np.arange(len(data.columns)) + 0.5)
ax.set_yticks(np.arange(len(data.columns)) + 0.5)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
ax.set_title("Conditional on day of year, at 15:00 local time")

plt.savefig("casestudy_pairwise_doy.png", transparent=True)


#
# STEP 6: Example scatter plot
#
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,6), sharey=True)
january_mask = afternoon_mask & (data.index.month == 1)
july_mask = afternoon_mask & (data.index.month == 7)

ax1.scatter(data.loc[january_mask, "WindDir"], data.loc[january_mask, "Temperature"], color="C0")
ax2.scatter(data.loc[july_mask, "WindDir"], data.loc[july_mask, "Temperature"], color="C1")

ax1.set_ylabel("Temperature")
ax1.set_xlabel("Wind direction")
ax2.set_xlabel("Wind direction")

ax1.set_xticks([0, 90, 180, 270, 360])
ax2.set_xticks([0, 90, 180, 270, 360])
ax1.set_xticklabels(["N", "E", "S", "W", "N"])
ax2.set_xticklabels(["N", "E", "S", "W", "N"])

ax1.set_title("January, 15:00 local time")
ax2.set_title("July, 15:00 local time")

plt.savefig("casestudy_wind_scatters.png", transparent=True)


#
# STEP 7: Time lags
#

# These are in decreasing order on the plot
covariates = ["Temperature", "DewPoint", "WindDir", "AirPressure", "WindSpeed"]

# Lag up to two days with 2-hour spacing
lags = np.arange(0, 2*24 + 1, 2)
temp = estimate_corr(data["Temperature"], data[covariates], lags,
    cond=data["DayOfYear"], mask=afternoon_mask)

# Plot the MI correlation coefficients as a line plot
fig, ax = plt.subplots(figsize=(8,6))
lines = ax.plot(temp)

# To make the results easier to interpret, display time points
# instead of lag values on the x axis. Only display every other time point.
ax.set_xticks(lags[::2])
ax.set_xticklabels([f"{(15-i) % 24}" for i in lags[::2]])
ax.invert_xaxis()

ax.set_xlabel("Covariate time")
ax.set_ylabel("MI correlation coefficient")
ax.set_title("Time dependency of temperature at 15:00 (conditional on DOY)")
ax.legend(lines, covariates)
ax.grid()

plt.savefig("casestudy_lags.png", transparent=True)


#
# STEP 8: Averaging
#

# Use two more masks, higher k, and average of the three runs
temp_12 = estimate_corr(data["Temperature"], data[covariates], lags, k=8,
    cond=data["DayOfYear"], mask=(data.index.hour == 12))
temp_13 = estimate_corr(data["Temperature"], data[covariates], lags, k=8,
    cond=data["DayOfYear"], mask=afternoon_mask)
temp_14 = estimate_corr(data["Temperature"], data[covariates], lags, k=8,
    cond=data["DayOfYear"], mask=(data.index.hour == 14))

temp_avg = (temp_12 + temp_13 + temp_14) / 3

# The same plotting code as above
fig, ax = plt.subplots(figsize=(8,6))
lines = ax.plot(temp_avg)

ax.set_xticks(lags[::2])
ax.set_xticklabels([f"{(15-i) % 24}" for i in lags[::2]])
ax.invert_xaxis()

ax.set_xlabel("Covariate time")
ax.set_ylabel("MI correlation coefficient")
ax.set_title("Time dependency of temperature centered at 15:00 (conditional on DOY)")
ax.legend(lines, covariates)
ax.grid()

plt.savefig("casestudy_lags_avg.png", transparent=True)
