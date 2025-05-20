# ---------------------------------------------
# replicate “Actual / Fitted / Residuals” plot
# ---------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA   

# To run: python week_3.py

# 1. read the data
file_path = "/Users/bazyliwidawski/Documents/Block 5, Year 3/Time Series Analysis/Python replication/USMSSALES.xlsx"
df = pd.read_excel(file_path)

# 2. convert “Quarter” strings → datetime index
df["Quarter"] = pd.PeriodIndex(df["Quarter"], freq="Q").to_timestamp()
df = df.set_index("Quarter")

# 3. linear-trend regression:  SALES_t = α + β·t + ε_t
t = np.arange(len(df))                     # 0, 1, 2, …
X = sm.add_constant(t)
ols_res   = sm.OLS(df["SALES"], X).fit()

df["Fitted"]   = ols_res.fittedvalues
df["Residual"] = ols_res.resid

# 4. build the dual-axis plot
fig, ax_left = plt.subplots(figsize=(12, 6))

# left axis: residuals
ax_left.plot(df.index, df["Residual"],
             color="tab:blue", linewidth=1.4, label="Residual")
ax_left.axhline(0, color="gray", linewidth=0.8)
ax_left.set_ylabel("Residual", color="tab:blue")
ax_left.tick_params(axis="y", labelcolor="tab:blue")
ax_left.set_ylim(-0.2, 0.25)
ax_left.grid(axis="y", linestyle="--", alpha=0.3)

# right axis: actual & fitted log-levels
ax_right = ax_left.twinx()
ax_right.plot(df.index, df["SALES"],
              color="sienna", linewidth=1.2, label="Actual")
ax_right.plot(df.index, df["Fitted"],
              color="seagreen", linewidth=1.2, label="Fitted")
ax_right.set_ylabel("Log US Manufacturing & Mining Sales",
                    color="seagreen")
ax_right.tick_params(axis="y", labelcolor="seagreen")
ax_right.set_ylim(df["SALES"].min()-0.1, df["SALES"].max()+0.1)

# common legend underneath
lines, labels = [], []
for ax in (ax_left, ax_right):
    lns, lbs = ax.get_legend_handles_labels()
    lines.extend(lns); labels.extend(lbs)
ax_right.legend(lines, labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=3, frameon=False)

ax_left.set_xlim(df.index.min(), df.index.max())
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

####################################################
# 4. Augmented Dickey-Fuller test (level, trend+intercept, BIC, maxlag=14) => IC can be changed ! 
adf_stat, pval, used_lag, nobs, crit_vals, icbest = adfuller(
    df["SALES"],
    maxlag=14,
    regression='ct',     # constant + trend
    autolag='AIC'
)

print("Augmented Dickey–Fuller test (trend & intercept)")
print(f"  ADF statistic : {adf_stat:8.4f}")
print(f"  p-value       : {pval:8.4f}")
print(f"  used lag      : {used_lag}")
print("  critical values:")
for k, v in crit_vals.items():
    print(f"     {k:>4} : {v:8.4f}")
print(f"  information criterion (BIC) choice : {icbest:8.4f}\n")

######################################################################
# fail to reject ADF => take the first diff of the data 
df = pd.read_excel(
    "/Users/bazyliwidawski/Documents/Block 5, Year 3/Time Series Analysis/"
    "Python replication/USMSSALES.xlsx"
)
df["Quarter"] = (pd.PeriodIndex(df["Quarter"], freq="Q")
                   .to_timestamp(how="e"))           # quarter-END timestamps
df = df.set_index("Quarter").asfreq("QE-DEC")        # silence future warnings
y  = df["SALES"]

# ────────────────────────────────────────────────────────────────
# 2│  build lagged regressors + linear time trend
# ────────────────────────────────────────────────────────────────
t = np.arange(len(y))                                # 0,1,2,…
lags = pd.concat([y.shift(i) for i in range(1, 4)], axis=1)
lags.columns = ["lag1", "lag2", "lag3"]

data = (pd.concat([y,
                   pd.Series(t, index=y.index, name="trend"),
                   lags],
                  axis=1)
          .dropna())                                 # lose first 3 rows

# ────────────────────────────────────────────────────────────────
# 3│  first 20 years (80 obs) → estimation sample
# ────────────────────────────────────────────────────────────────
train_n = 80
X_train = sm.add_constant(data.iloc[:train_n][["trend", "lag1", "lag2", "lag3"]])
y_train = data.iloc[:train_n]["SALES"]

ols_res = sm.OLS(y_train, X_train).fit()
c0, c1, ϕ1, ϕ2, ϕ3 = ols_res.params
σ2 = ols_res.scale                                 # residual variance
print(ols_res.summary())

# (X'X)⁻¹ for the variance-inflation term
XtX_inv = np.linalg.inv(X_train.to_numpy().T @ X_train.to_numpy())

# ────────────────────────────────────────────────────────────────
# 4│  one-step recursive forecasts for the test period
# ────────────────────────────────────────────────────────────────
test = data.iloc[train_n:]                          # after the first 80 obs
trend_pred = test["trend"] + 1                      # t+1 trend
fcst = (c0
        + c1 * trend_pred
        + ϕ1 * test["lag1"]
        + ϕ2 * test["lag2"]
        + ϕ3 * test["lag3"])

# ── build the same design matrix used in forecasting
X_pred = np.column_stack([np.ones(len(test)),
                          trend_pred,
                          test[["lag1", "lag2", "lag3"]]])

# full forecast s.e. = √[ σ² · (1 + xᵀ(X'X)⁻¹x) ]
se_full = np.sqrt(σ2 * (1 + np.sum((X_pred @ XtX_inv) * X_pred, axis=1)))

upper = fcst + 1.96 * se_full
lower = fcst - 1.96 * se_full
y_test = test["SALES"]

# ────────────────────────────────────────────────────────────────
# 5│  accuracy measures
# ────────────────────────────────────────────────────────────────
err  = y_test - fcst
rmse = np.sqrt(np.mean(err**2))
mae  = np.mean(np.abs(err))
mape = np.mean(np.abs(err / y_test))
smape= np.mean(2 * np.abs(err) / (np.abs(y_test) + np.abs(fcst)))
u1   = rmse / (np.sqrt((fcst**2).mean()) + np.sqrt((y_test**2).mean()))
u2   = rmse / np.sqrt(np.mean((y_test - y.shift(1).loc[y_test.index])**2))
bias = ((fcst.mean() - y_test.mean())**2) / rmse**2
var  = ((fcst.std()  - y_test.std()) **2) / rmse**2
cov  = 1 - bias - var

# ────────────────────────────────────────────────────────────────
# 6│  plot
# ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(fcst,   lw=1.4, color="steelblue", label="YF_AR3T")
ax.plot(y_test, lw=1.2, color="seagreen",  label="Actuals")
ax.plot(upper,  ls="--", color="peru",  lw=0.9, label="95 % PI")
ax.plot(lower,  ls="--", color="peru",  lw=0.9)
ax.legend(frameon=False, ncol=3)
ax.set_ylabel("Log level"); ax.grid(axis="x", ls=":", alpha=.4)
fig.autofmt_xdate()

stats = [
    "Forecast:  YF_AR3T",
    "Actual:    Y",
    f"Forecast sample: {y_test.index[0].strftime('%YQ%q')} "
    f"{y_test.index[-1].strftime('%YQ%q')}",
    f"Included observations: {len(y_test)}",
    f"Root Mean Squared Error      {rmse:10.6f}",
    f"Mean Absolute Error          {mae:10.6f}",
    f"Mean Abs. Percent Error      {mape:10.6f}",
    f"Theil Inequality Coef.       {u1:10.6f}",
    f"  Bias Proportion            {bias:10.6f}",
    f"  Variance Proportion        {var:10.6f}",
    f"  Covariance Proportion      {cov:10.6f}",
    f"Theil U2 Coefficient         {u2:10.6f}",
    f"Symmetric MAPE               {smape:10.6f}",
]
ax.text(1.02, 0.98, "\n".join(stats),
        transform=ax.transAxes, va="top", ha="left",
        family="monospace", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=.9))

plt.tight_layout()
plt.show()