# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA   

# import the dataset 
df = pd.read_excel("/Users/bazyliwidawski/Documents/Block 5, Year 3/Time Series Analysis/Python replication/USCC.xlsx")
df["Quarter"] = pd.PeriodIndex(df["Quarter"], freq="Q").to_timestamp()
df = df.set_index("Quarter")

print(df.head())

# Initial plot the data
plt.figure(figsize=(10, 6))
plt.plot(df.index[-100:], df["USCC"][-100:], label="USCC", color="orange")
plt.plot(df.index[-100:], df["USCCSA"][-100:], label="USCCSA", color="blue")
plt.title("USCC Time Series")
plt.xlabel("Date")
plt.ylabel("USCC")
plt.legend()
plt.show()

# convert to the log scale 
plt.figure(figsize=(10, 6))
plt.plot(df.index[-100:], np.log(df["USCC"][-100:]), label="USCC", color="orange")
plt.plot(df.index[-100:], np.log(df["USCCSA"][-100:]), label="USCCSA", color="blue")
plt.title("USCC Time Series (Log Scale)")
plt.xlabel("Date")
plt.ylabel("Log USCC")
plt.legend()
plt.show()

#############################################################################
# ADF TEST ON GROWTH RATES

# deviation for the trend already in the week_3.py code
# 2) Compute log‐levels, then first differences (growth rates)
df["log_USCC"]   = np.log(df["USCC"])
df["log_USCCSA"] = np.log(df["USCCSA"])
df["dlog_USCC"]   = df["log_USCC"].diff() * 100      # Δlog × 100
df["dlog_USCCSA"] = df["log_USCCSA"].diff() * 100    # Δlog × 100

# 3) Drop the first row (NaN) so we can run ADF
df = df.dropna(subset=["dlog_USCC", "dlog_USCCSA"])

# 4) Define a helper that runs ADF with constant+trend
def adf_with_trend(series, max_lag=14):
    series = series.dropna()
    adf_stat, pval, used_lag, nobs, crit_vals, icbest = adfuller(
        series,
        maxlag=max_lag,
        regression="ct",    # include constant + linear trend
        autolag="AIC"
    )
    return adf_stat, pval, used_lag, crit_vals, icbest

# 5) Run on each growth‐rate series
for name in ["dlog_USCC", "dlog_USCCSA"]:
    adf_stat, pval, used_lag, crit_vals, icbest = adf_with_trend(df[name])
    print(f"\n======== ADF Results for {name} ========")
    print(f"ADF statistic   = {adf_stat:.5f}")
    print(f"p-value         = {pval:.5f}")
    print(f"used lag        = {used_lag}")
    print("critical values :")
    for lvl, val in crit_vals.items():
        print(f"   {lvl} → {val:.5f}")
    print(f"best IC (AIC)   = {icbest:.5f}")

###########################################################################
# EXTRACT THE ROOTS OF THE AR(4) MODEL with SEASONAL DUMMY VARIABLES

y = df["USCC"]

df["D1"] = (df.index.quarter == 1).astype(int)
df["D2"] = (df.index.quarter == 2).astype(int)
df["D3"] = (df.index.quarter == 3).astype(int)
df["D4"] = (df.index.quarter == 4).astype(int)

df["T"] = np.arange(len(df))

exog = df[["D1", "D2", "D3", "D4", "T"]]


model = ARIMA(
    endog=y, 
    exog=exog,
    order=(4, 0, 0),
    trend="n"           # “n” means no additional constant; the dummies absorb the level.
)
res = model.fit()

print(res.summary())

# Note: in statsmodels ARIMA, the naming for AR coefficients is 'ar.L1', 'ar.L2', ...
phi1 = res.params["ar.L1"]
phi2 = res.params["ar.L2"]
phi3 = res.params["ar.L3"]
phi4 = res.params["ar.L4"]

# Build the characteristic polynomial coefficients [1, -φ1, -φ2, -φ3, -φ4]
poly_coeffs = [1.0, -phi1, -phi2, -phi3, -phi4]

# Solve for all (up to 4) complex roots of the polynomial
roots = np.roots(poly_coeffs)

print("\n=== ACTUAL AR(4) ROOTS AND THEIR MAGNITUDES ===")
for i, r in enumerate(roots, start=1):
    print(f"Root #{i:>d}: {r:.5f}   |  |root| = {np.abs(r):.5f}")


###########################################################################
# PLOT THE ROOTS ON THE COMPLEX PLANE

theta = np.linspace(0, 2*np.pi, 200)
x_circle = np.cos(theta)
y_circle = np.sin(theta)

plt.figure(figsize=(6, 6))
# Draw unit circle
plt.plot(x_circle, y_circle, color='gray', linestyle='--', label='Unit Circle')

# Plot each root
plt.scatter(roots.real, roots.imag, color='blue', s=50, zorder=5, label='AR Roots')

# Add axes through origin
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Labels, title, and formatting
plt.title("AR(4) Roots on the Complex Plane")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.grid(True, linestyle=':', alpha=0.5)
plt.gca().set_aspect('equal', 'box')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.legend(loc="upper right")
plt.show()