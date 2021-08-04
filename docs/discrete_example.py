from ennemi import estimate_entropy, estimate_mi, pairwise_mi
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd

rcParams["lines.markersize"] = 12

N = 200
week = np.arange(N)
rng = np.random.default_rng(1234)

# The weather is completely determined by temperature and air pressure
# NOTE: This is not a realistic weather model! :)
actual_temp = 15 + 5*np.sin(week / 8) + rng.normal(0, 3, N)
actual_press = 1000 + 30*np.sin(week / 3) + rng.normal(0, 4, N)
wind_dir = rng.choice(["N", "E", "S", "W"], N, p=[0.15, 0.15, 0.4, 0.3])

weather = np.full(N, "cloudy")
weather[(actual_press > 1015) & (actual_temp > 18)] = "clear"
weather[(weather=="cloudy") & (wind_dir=="W")] = "rainy"
weather[(weather=="cloudy") & (wind_dir=="S") & rng.choice([0,1], N)] = "rainy"

# The measurements for these are not accurate either
temp = np.round(actual_temp + rng.normal(0, 1, N))
press = np.round(actual_press + rng.normal(0, 1, N))

# Create a pandas data frame out of the measurements
data = pd.DataFrame({"Weather": weather, "Temp": temp, "Press": press, "Wind": wind_dir})
print("Sample of the data:")
print(data)

# Plot the weather for one "year"
for (forecast, marker, color) in [("cloudy", "$\u2601$", "gray"),
                                  ("clear", "$\u2600$", "orange"),
                                  ("rainy", "$\u2602$", "blue")]:
    plt.scatter(week[weather==forecast], temp[weather==forecast],
        marker=marker, color=color)

plt.title("Weather for Entropyville")
plt.xlabel("Week")
plt.ylabel("Temperature")
plt.xlim((0, 50))
plt.savefig("discrete_temp_weather.png", transparent=True)


#
# Fix-up step for pandas DataFrames
#
print("\nFix up data")

# Not the most optimal code, but sufficient in small example
data2 = data.drop(columns=["Weather", "Wind"])

data2["Wind"] = 0
data2.loc[data["Wind"] == "E", "Wind"] = 1
data2.loc[data["Wind"] == "S", "Wind"] = 2
data2.loc[data["Wind"] == "W", "Wind"] = 3

data2["Weather"] = 0
data2.loc[data["Weather"] == "cloudy", "Weather"] = 1
data2.loc[data["Weather"] == "clear", "Weather"] = 2

print(data2)
print(data2.dtypes)



#
# Correlation between continuous variables and weather
#

print("\MI between continuous variables and weather")
print(estimate_mi(data2["Weather"], data2[["Temp", "Press"]], discrete_y=True))
print("Entropy of Weather")
print(estimate_entropy(data2["Weather"], discrete=True))


#
# Conditioning on temperature
#

print("\nConditioned on temperature")
print(estimate_mi(data2["Weather"], data2["Press"], cond=data2["Temp"], discrete_y=True))


#
# Wind
#
print("\nMI between wind and weather")
print(estimate_mi(data2["Weather"], data2["Wind"], discrete_y=True, discrete_x=True))
print("MI between wind and continuous variables")
print(estimate_mi(data2["Wind"], data2[["Temp", "Press"]], discrete_y=True))

# Uncomment to get a warning
#print("\nConditioned on temperature and pressure")
#print(estimate_mi(data2["Weather"], data2["Wind"],
#      cond=data2[["Temp","Press"]], discrete_y=True, discrete_x=True))


#
# Pairwise MI
#
print("\nPairwise MI")
print(pairwise_mi(data2, discrete=[False, False, True, True]))
