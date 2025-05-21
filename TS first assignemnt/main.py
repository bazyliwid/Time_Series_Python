# import necesary libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_process import arma_acf
from statsmodels.stats.stattools import jarque_bera
import math
from itertools import groupby
from operator import itemgetter

# To run: python main.py

# import the excel file
data = pd.read_excel('/Users/bazyliwidawski/Documents/Block 5, Year 3/Time Series Analysis/Python replication/TS first assignemnt/RSFHFS.xlsx',header=0,parse_dates=["QUARTER"],index_col="QUARTER")

print(data.columns)


#################################################################################################################
# TREND INVESTIGATION

# check the presence of the trend:
# Create time trend: numerical index
time = range(len(data))
X = sm.add_constant(time)  # Adds intercept

# Regress on linear trend
model = sm.OLS(data['RSFHFS'], X)
results = model.fit()

print(results.summary())

# Fitted values
trend_line = results.fittedvalues

# Plot original series and trend
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['RSFHFS'], label='Original Series', color='purple')
plt.plot(data.index, trend_line, label='Linear Trend', color='red', linestyle='--')
plt.fill_between(data.index, data['RSFHFSN'].min(), data['RSFHFSN'].max(),
                where=(data['DREC']==1),
                color='tab:grey', 
                alpha=0.3,
                label='Event period')
plt.title('US Initial Claims for Unemployment Insurance with Linear Trend')
plt.xlabel('Date')
plt.ylabel('Claims')
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.show()

#################################################################################################################

# check the presence of the trend:
# Create time trend: numerical index
time = range(len(data))
X = sm.add_constant(time)  # Adds intercept

# Regress on linear trend
model = sm.OLS(data['RSFHFSN'], X)
results_ns = model.fit()

print(results_ns.summary())

# Fitted values
trend_line_ns = results_ns.fittedvalues

# Plot original series and trend
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['RSFHFSN'], label='Original Series', color='purple')
plt.plot(data.index, trend_line_ns, label='Linear Trend', color='red', linestyle='--')
plt.fill_between(data.index, data['RSFHFSN'].min(), data['RSFHFSN'].max(),
                where=(data['DREC']==1),
                color='tab:grey', 
                alpha=0.3,
                label='Event period')
plt.title('US Initial Claims for Unemployment Insurance with Linear Trend')
plt.xlabel('Date')
plt.ylabel('Claims')
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data.index, data['RSFHFS'].diff(), label='Original Series', color='purple')
plt.fill_between(data.index, data['RSFHFS'].diff().min(), data['RSFHFS'].diff().max(),
                where=(data['DREC']==1),
                color='tab:grey', 
                alpha=0.3,
                label='Event period')
plt.title('US Initial Claims for Unemployment Insurance with Linear Trend')
plt.xlabel('Date')
plt.ylabel('Claims')
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.show()


#################################################################################################################
# SEASON INVESTIGATION 
data = pd.read_excel('/Users/bazyliwidawski/Documents/Block 5, Year 3/Time Series Analysis/Python replication/TS first assignemnt/RSFHFS.xlsx',header=0)

quarterly_data = pd.DataFrame()

for i in range(1,5):
    
    #quarter_data = data.loc[data["Quarter"].apply(lambda x: "Q{}".format(i) in str(x)), "US initial claims for unemployment insurance"]
    #quarter_data = data.loc["Q{}".format(i) in str(data["Quarter"]), "US initial claims for unemployment insurance"]
    quarter_data = (data["RSFHFSN"][data["QUARTER"].str.contains(f"Q{i}", na=False)])
    quarter_data = quarter_data.reset_index(drop=True)
    quarterly_data["Q{}".format(i)] = quarter_data 
    
fig, ax = plt.subplots()
ax.plot(
        quarterly_data.index, 
        quarterly_data["Q1"],
        linestyle="-",
        label="Q1"
        )
ax.plot(
        quarterly_data.index, 
        quarterly_data["Q2"],
        linestyle="-",
        label="Q2"
        )
ax.plot(
        quarterly_data.index, 
        quarterly_data["Q3"],
        linestyle="-",
        label="Q3"
        )
ax.plot(
        quarterly_data.index, 
        quarterly_data["Q4"],
        linestyle="-",
        label="Q4"
        )
ax.set_title("Vector of Quarters Plot")
ax.set_xlabel("Year")
ax.set_ylabel("Value")
ax.legend()                    
plt.tight_layout()
plt.show()

data[['Year', 'QuarterNum']] = data['QUARTER'].str.extract(r'(\d{4})Q([1-4])')
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
data['QuarterNum'] = pd.to_numeric(data['QuarterNum'], errors='coerce')

# Drop rows with missing values in key columns
data = data.dropna(subset=['Year', 'QuarterNum', 'RSFHFSN'])

# Manually create dummy variables (Q1 is baseline)
data['Q1'] = (data['QuarterNum'] == 1).astype(int)
data['Q2'] = (data['QuarterNum'] == 2).astype(int)
data['Q3'] = (data['QuarterNum'] == 3).astype(int)
data['Q4'] = (data['QuarterNum'] == 4).astype(int)

# Use all four dummies, but drop intercept
X = data[['Q1','Q2', 'Q3', 'Q4']]

# Fit model with no intercept
model = sm.OLS(data['RSFHFSN'], X).fit()

print(model.summary())

# F-test for the coefficients 
wald_test = model.f_test(["Q1 - Q3 = 0", "Q1 - Q4 = 0",'Q1 - Q2 = 0'])
print(wald_test)

#################################################################################################################
# ABBERANT INVESTIGATION 

# indentfied and elaborated (crises)

#################################################################################################################
# HETEROGENUITY INVESTIGATION 

diff = data['RSFHFS'].diff().dropna()

window = 20
rolling_std = diff.rolling(window=window).std()

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(rolling_std.index, rolling_std, color='blue', label=f'{window}-Q Rolling Std of ΔRSFHFS')

drec = data['DREC'].reindex(rolling_std.index)
ax.fill_between(
    rolling_std.index,
    0, rolling_std.max(),
    where=(drec == 1),
    color='grey',
    alpha=0.3,
    label='Event period'
)

ax.set_title(f'Rolling {window}-Quarter Std Dev of Δ(RSFHFS)')
ax.set_xlabel('Date')
ax.set_ylabel('Std Dev')
ax.legend()
# ax.grid(True)
plt.tight_layout()
plt.show()

#################################################################################################################
# NON-LINEARITY 

X = pd.concat(
        [data['DREC'],      
         1 - data['DREC']],    
        axis=1
     )
X.columns = ['DREC', 'EXP']

y_diff = data['RSFHFS'].diff().dropna()  

X_diff = X.loc[y_diff.index]

model_diff = sm.OLS(y_diff, X_diff).fit()
print(model_diff.summary())

wald_test = model_diff.f_test("DREC - EXP = 0") 
print(wald_test)
#################################################################################################################
# MAT BLAME 

# The code below generates the empirical results for the ACF, PACF, and Ljung–Box test on the truncated sample

# select the series
data_idx = data.set_index(pd.PeriodIndex(data['QUARTER'], freq='Q'))

# now you can slice by quarter‐strings
filtered = data_idx.loc['1992Q2':'2009Q4'].reset_index(drop=True)

ts1 = np.log(filtered['RSFHFS']).diff().dropna()
# choose max lag
nlags = 20

# 1) compute ACF & its 95% CI
acf_vals, acf_confint = acf(ts1, nlags=nlags, alpha=0.05, fft=False)

# 2) compute PACF & its 95% CI
pacf_vals, pacf_confint = pacf(ts1, nlags=nlags, alpha=0.05)

# 3) Ljung–Box test up to each lag
#    `return_df=True` gives a DataFrame with columns ['lb_stat', 'lb_pvalue']
lb_test = acorr_ljungbox(ts1, lags=nlags, return_df=True)

# assemble into one table
lags = np.arange(1, nlags+1)  # skip lag=0 for display
table = pd.DataFrame({
    'lag':            lags,
    'ACF':            acf_vals[1:],           # drop lag=0
    'ACF_lower95':    acf_confint[1:, 0],
    'ACF_upper95':    acf_confint[1:, 1],
    'PACF':           pacf_vals[1:],          # drop lag=0
    'PACF_lower95':   pacf_confint[1:, 0],
    'PACF_upper95':   pacf_confint[1:, 1],
    'Q-Stat':         lb_test['lb_stat'].values,
    'Prob':           lb_test['lb_pvalue'].values,
})

print(table.to_string(index=False))

fig, axes = plt.subplots(2, 1, figsize=(8, 6))
sm.graphics.tsa.plot_acf(ts1, lags=nlags, ax=axes[0])
sm.graphics.tsa.plot_pacf(ts1, lags=nlags, ax=axes[1])
plt.tight_layout()
plt.savefig('/Users/bazyliwidawski/Documents/Block 5, Year 3/Time Series Analysis/Python replication/ACF_PACF_PARTIAL.png') 

# The code below generates the empirical results for the ACF, PACF, and Ljung–Box test on the full sample

ts2 = np.log(data['RSFHFS']).diff().dropna()
# choose max lag
nlags = 20

# 1) compute ACF & its 95% CI
acf_vals, acf_confint = acf(ts2, nlags=nlags, alpha=0.05, fft=False)

# 2) compute PACF & its 95% CI
pacf_vals, pacf_confint = pacf(ts2, nlags=nlags, alpha=0.05)

# 3) Ljung–Box test up to each lag
#    `return_df=True` gives a DataFrame with columns ['lb_stat', 'lb_pvalue']
lb_test = acorr_ljungbox(ts2, lags=nlags, return_df=True)

# assemble into one table
lags = np.arange(1, nlags+1)  # skip lag=0 for display
table = pd.DataFrame({
    'lag':            lags,
    'ACF':            acf_vals[1:],           # drop lag=0
    'ACF_lower95':    acf_confint[1:, 0],
    'ACF_upper95':    acf_confint[1:, 1],
    'PACF':           pacf_vals[1:],          # drop lag=0
    'PACF_lower95':   pacf_confint[1:, 0],
    'PACF_upper95':   pacf_confint[1:, 1],
    'Q-Stat':         lb_test['lb_stat'].values,
    'Prob':           lb_test['lb_pvalue'].values,
})

print(table.to_string(index=False))

fig, axes = plt.subplots(2, 1, figsize=(8, 6))
sm.graphics.tsa.plot_acf(ts2, lags=nlags, ax=axes[0])
sm.graphics.tsa.plot_pacf(ts2, lags=nlags, ax=axes[1])
plt.tight_layout()
plt.savefig('/Users/bazyliwidawski/Documents/Block 5, Year 3/Time Series Analysis/Python replication/ACF_PACF_FULL.png') 

filtered2 = data_idx.loc['1992Q2':'2009Q4']
ts3 = np.log(filtered2['RSFHFS']).diff().dropna()

# The code below estimates AR(p) and MA(q) models for p,q=0,1,2,3 and records the corresponding AIC and BIC values


model_IC_score = pd.DataFrame(columns=['MA_AIC', 'MA_BIC', 'AR_AIC', 'AR_BIC'])

for i in range(4):
    dataf = ts3.loc['1993Q1':'2009Q4']
    ar_model = sm.tsa.ARIMA(dataf, order=(i,0,0)).fit()
    ma_model = sm.tsa.ARIMA(dataf, order=(0,0,i)).fit()
    print(ar_model.summary())
    print(ma_model.summary())
    model_IC_score.loc[i] = [ma_model.aic, ma_model.bic, ar_model.aic, ar_model.bic]

print(model_IC_score)

dataf = ts3.loc['1993Q1':'2009Q4']
ar2_model = sm.tsa.ARIMA(dataf, order=(2,0,0)).fit()
r_1, r_2 = ar2_model.arroots
print(r_1, r_2)
phi1, phi2 = ar2_model.params['ar.L1'], ar2_model.params['ar.L2']
theoretical_acf = arma_acf([1, -phi1, -phi2], [1], lags=20)
print(theoretical_acf)

############################################################################
lags       = 20          # number of lags to display
series     = dataf       # <- the log-diff series you gave to ar2_model
bar_width  = 0.35        # skinny bar width

# ---------------------------------------------------------------
# 2) empirical ACF / PACF
# ---------------------------------------------------------------
emp_acf  = acf(series,  nlags=lags, fft=False)
emp_pacf = pacf(series, nlags=lags, method='ywmle')

# ---------------------------------------------------------------
# 3) theoretical ACF / PACF of AR(2)
# ---------------------------------------------------------------
phi1, phi2 = ar2_model.params[['ar.L1', 'ar.L2']]

# ask for one extra lag so theo_acf[1:] has exactly `lags` points
theo_acf = arma_acf(ar=[1, -phi1, -phi2], ma=[1], lags=lags + 1)

theo_pacf = np.zeros(lags + 1)
theo_pacf[1] = phi1
theo_pacf[2] = phi2 / (1 - phi1**2)   # exact AR(2) PACF at lag 2
# lags ≥ 3 stay at 0

# ---------------------------------------------------------------
# 4) plotting
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
lags_idx = np.arange(1, lags + 1)

# ---------- ACF panel ------------------------------------------
axes[0].bar(lags_idx, emp_acf[1:], width=bar_width,
            color='steelblue', label='Empirical')

axes[0].plot(lags_idx, theo_acf[1:], color='red',
             linewidth=1.8, label='Theoretical')

axes[0].set_ylim(-0.4, 1.2)
axes[0].set_xlim(0.5, lags + 0.5)
axes[0].set_ylabel('Autocorrelation')
axes[0].legend(loc='upper right')

# ---------- PACF panel -----------------------------------------
axes[1].bar(lags_idx, emp_pacf[1:], width=bar_width,
            color='steelblue', label='Empirical')

# red theoretical PACF line:
# • centre of bar 1  ➔  centre of bar 2  ➔  0 at lag 3  ➔  stays at 0
x_theo = np.concatenate(([1, 2], np.arange(3, lags + 1)))
y_theo = np.concatenate((
            [theo_pacf[1], theo_pacf[2]],
            np.zeros(lags - 2)
         ))
axes[1].plot(x_theo, y_theo,
             color='red', linewidth=1.8, label='Theoretical')

axes[1].set_ylim(-0.5, 1.0)
axes[1].set_xlim(0.5, lags + 0.5)
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('Partial autocorrelation')
axes[1].legend(loc='upper right')

# ---------- output ---------------------------------------------
plt.tight_layout()
plt.savefig('ACF_PACF_theoretical_vs_empirical_AR2.png', dpi=300)
plt.show()

################################################################################################################################################
residuals = ar2_model.resid
residuals_squared = np.square(ar2_model.resid)


acf_vals, acf_confint = acf(residuals, nlags=20, alpha=0.05, fft=False)
emp_pacf = pacf(residuals, nlags=lags, method='ywmle')
lb_test = acorr_ljungbox(residuals, lags=lags, return_df=True)


table = pd.DataFrame({
    'lag':            lags,
    'ACF':            acf_vals[1:],           # drop lag=0
    'ACF_lower95':    acf_confint[1:, 0],
    'ACF_upper95':    acf_confint[1:, 1],
    'PACF':           pacf_vals[1:],          # drop lag=0
    'PACF_lower95':   pacf_confint[1:, 0],
    'PACF_upper95':   pacf_confint[1:, 1],
    'Q-Stat':         lb_test['lb_stat'].values,
    'Prob':           lb_test['lb_pvalue'].values,
})

print(table)

acf_vals, acf_confint = acf(residuals_squared, nlags=20, alpha=0.05, fft=False)
emp_pacf = pacf(residuals_squared, nlags=lags, method='ywmle')
lb_test = acorr_ljungbox(residuals_squared, lags=lags, return_df=True)


table = pd.DataFrame({
    'lag':            lags,
    'ACF':            acf_vals[1:],           # drop lag=0
    'ACF_lower95':    acf_confint[1:, 0],
    'ACF_upper95':    acf_confint[1:, 1],
    'PACF':           pacf_vals[1:],          # drop lag=0
    'PACF_lower95':   pacf_confint[1:, 0],
    'PACF_upper95':   pacf_confint[1:, 1],
    'Q-Stat':         lb_test['lb_stat'].values,
    'Prob':           lb_test['lb_pvalue'].values,
})

print(table)

def test_normality(growthRate):
    mean = growthRate.mean()
    median = growthRate.median()
    mn, mx = growthRate.min(), growthRate.max()
    std = growthRate.std()
    skew = growthRate.skew()
    kurt = growthRate.kurtosis()
    jb_stat, jb_pvalue, jb_skew, jb_kurt = jarque_bera(growthRate)

    print(jarque_bera(growthRate))

    fig, (ax_hist, ax_stats) = plt.subplots(1, 2, figsize=(12, 5),
                                            gridspec_kw={'width_ratios': [3, 1]})

    # Histogram with density=True to match the PDF scale
    count, bins, ignored = ax_hist.hist(growthRate, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7, label='Histogram')

    # Normal distribution curve
    x = np.linspace(min(bins), max(bins), 1000)
    normal_pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    ax_hist.plot(x, normal_pdf, 'r-', label='Normal')

    ax_hist.set_xlabel('Growth rate (%)')
    ax_hist.set_ylabel('Density')
    ax_hist.legend()

    # Statistics box
    ax_stats.axis('off')
    text = (
        f'N         {len(growthRate)}\n'
        f'Mean      {mean:8.4f}\n'
        f'Median    {median:8.4f}\n'
        f'Max       {mx:8.4f}\n'
        f'Min       {mn:8.4f}\n'
        f'Std. Dev. {std:8.4f}\n'
        f'Skewness  {jb_skew:8.4f}\n'
        f'Kurtosis  {jb_kurt:8.4f}\n'
        f'Jarque-Bera  {jb_stat:8.2f}\n'
        f'Prob(JB)     {jb_pvalue:.6f}'
    )
    ax_stats.text(0.05, 0.95, text, transform=ax_stats.transAxes,
                  fontsize=10, va='top', family='monospace')

    plt.tight_layout()
    plt.show()

# Test for normality for the whole time series 
test_normality(residuals)

################################################################################################################################################
y = ts2                                   # log-diff series (PeriodIndex)

# 1) lags
lags = pd.concat([y.shift(i) for i in range(1, 3)], axis=1)
lags.columns = ["lag1", "lag2"]

# 2) assemble DataFrame & drop *all* duplicates once
full = (pd.concat([y, lags], axis=1)
          .dropna())                       # rows with all vars present
full = full.loc[~full.index.duplicated()]  # ← critical: unique index

# 3) AR(2) coefficients + variance
c0  = ar2_model.params["const"]
φ1  = ar2_model.params["ar.L1"]
φ2  = ar2_model.params["ar.L2"]
σ2  = np.std(ar2_model.resid)

# 4) estimation window 1993Q1–2009Q4
train = full.loc["1993Q1":"2009Q4"]
X_train = sm.add_constant(train[["lag1", "lag2"]], has_constant="add")
XtX_inv = np.linalg.inv(X_train.T @ X_train)

# 5) test window 2010Q1–2024Q4  (fallback if that slice is empty)
test = full.loc["2010Q1":"2024Q4"]
if test.empty:
    test = full.loc[train.index[-1] + 1:]          # everything after train

test = test.loc[~test.index.duplicated()]          # ensure unique again

# 6) forecasts & 95 % PI
fcst  = c0 + φ1*test["lag1"] + φ2*test["lag2"]
X_pred = sm.add_constant(test[["lag1", "lag2"]], has_constant="add").to_numpy()
quad   = (X_pred @ XtX_inv) * X_pred
se_full= np.sqrt(np.maximum(0, σ2 * (1 + quad.sum(axis=1))))  # no NaNs

upper = fcst + 1.96*se_full
lower = fcst - 1.96*se_full
y_test= test["RSFHFS"]

# ------------ accuracy metrics ----------------------------------
err   = y_test - fcst
rmse  = np.sqrt(np.mean(err**2))
mae   = np.mean(np.abs(err))
mape  = np.mean(np.abs(err / y_test))
smape = np.mean(2*np.abs(err)/(np.abs(y_test)+np.abs(fcst)))
u1    = rmse / (np.sqrt((fcst**2).mean()) + np.sqrt((y_test**2).mean()))
u2    = rmse / np.sqrt(np.mean((y_test - y.shift(1).loc[y_test.index])**2))
bias  = ((fcst.mean() - y_test.mean())**2) / rmse**2
var   = ((fcst.std()  - y_test.std()) **2) / rmse**2
cov   = 1 - bias - var

def qlabel(idx):
    try:    return idx.strftime("%YQ%q")
    except: return str(idx)

# ------------ plot ----------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(fcst,   lw=1.4, color="steelblue", label="YF_AR2")
ax.plot(y_test, lw=1.2, color="seagreen",  label="Actuals")
ax.plot(upper,  ls="--", color="peru",     lw=0.9, label="95 % PI")
ax.plot(lower,  ls="--", color="peru",     lw=0.9)
ax.legend(frameon=False, ncol=3)
ax.set_ylabel("RSFHFS level (log)")
ax.grid(axis="x", ls=":", alpha=.4)
fig.autofmt_xdate()

stats = [
    "Forecast:  YF_AR2",
    "Actual:    Y",
    f"Forecast sample: {qlabel(y_test.index[0])} – {qlabel(y_test.index[-1])}",
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