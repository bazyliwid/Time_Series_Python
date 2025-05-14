import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as stats

# thats for me 
# to activate: conda activate /opt/anaconda3/envs/test
# to run: python week_2.py

############################################################################################################################################################################
# ACF & PACF & Ljung-Box test
############################################################################################################################################################################

# Load the data (paste ur own path)
data_1 = pd.read_excel('/Users/bazyliwidawski/Documents/Block 5, Year 3/Time Series Analysis/Python replication/USUNINSCE.xlsx', parse_dates=['Quarter'], index_col='Quarter')

# select the series
ts = data_1['US initial claims for unemployment insurance'].dropna()

# choose max lag
nlags = 20

# 1) compute ACF & its 95% CI
acf_vals, acf_confint = acf(ts, nlags=nlags, alpha=0.05, fft=False)

# 2) compute PACF & its 95% CI
pacf_vals, pacf_confint = pacf(ts, nlags=nlags, alpha=0.05)

# 3) Ljung–Box test up to each lag
#    `return_df=True` gives a DataFrame with columns ['lb_stat', 'lb_pvalue']
lb_test = acorr_ljungbox(ts, lags=nlags, return_df=True)

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

# 4) optionally plot
fig, axes = plt.subplots(2, 1, figsize=(8, 6))
sm.graphics.tsa.plot_acf(ts, lags=nlags, ax=axes[0])
sm.graphics.tsa.plot_pacf(ts, lags=nlags, ax=axes[1])
plt.tight_layout()
plt.savefig('/Users/bazyliwidawski/Documents/Block 5, Year 3/Time Series Analysis/Python replication/ACF_PACF.png') 

# => clearly a AR(3) model, just run the regression 
# => ARIMA(3,0,0) model , investigate the properties of the residuals 

# ===================================================================
# 1) Fit an AR(3) with constant
# ===================================================================
# you can also use AutoReg if you prefer:
# from statsmodels.tsa.ar_model import AutoReg
# model = AutoReg(ts, lags=3, trend='c').fit()

model = sm.tsa.ARIMA(ts, order=(3,0,0)).fit()
print(model.summary())

# ===================================================================
# 2) Build a results DataFrame
# ===================================================================
df = ts.to_frame(name='Actual')
df['Fitted']   = model.fittedvalues
df['Residual'] = model.resid

# extract Year & Quarter from the DatetimeIndex
df['Year']       = df.index.year
df['QuarterNum'] = df.index.quarter
df['TimeLabel']  = df['Year'].astype(str) + 'Q' + df['QuarterNum'].astype(str)

# (optional) if you need separate Year/Quarter columns later:
# df = df.reset_index()

# ===================================================================
# 3) Plot Actual vs Fitted and Residuals
# ===================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Top: Actual vs Fitted
ax1.plot(df.index, df['Actual'], label='Actual', linewidth=1)
ax1.plot(df.index, df['Fitted'], label='Fitted (AR(3))', linewidth=1)
ax1.set_title('Actual vs Fitted')
ax1.set_ylabel('Claims')
ax1.legend()

# Bottom: Residuals
ax2.plot(df.index, df['Residual'], label='Residuals', linewidth=0.9)
ax2.axhline(0, color='gray', linestyle='--', linewidth=0.7)
ax2.set_title('Residuals of AR(3) Model')
ax2.set_ylabel('Residual')
ax2.set_xlabel('Quarter')
ax2.legend()

plt.tight_layout()
plt.show()

############################################################################################################################################################################
# FORECASTING PART 
############################################################################################################################################################################

df = pd.read_excel(
    '/Users/bazyliwidawski/Documents/Block 5, Year 3/'
    'Time Series Analysis/Python replication/USUNINSCE.xlsx',
    dtype={'Quarter': str}
)
df['Quarter'] = df['Quarter'].str.strip()
df.index = pd.PeriodIndex(df['Quarter'], freq='Q')
df = df.drop(columns='Quarter')

ts = df['US initial claims for unemployment insurance'].asfreq('Q')

# --- 1) set your split points -------------------------------
train_end      = '1986Q4'    # estimation sample ends here
forecast_start = '1987Q1'
forecast_end   = '2019Q4'

# quarterly periods for forecasting
fc_periods = ts.loc[forecast_start:forecast_end].index

# placeholders
forecasts = []
lower95   = []
upper95   = []
ses       = []
actuals   = []

# --- 2) rolling one‐step loop ---------------------------------
for period in fc_periods:
    # build “history” up to period-1
    history = ts[: period - 1]  

    # re-estimate AR(3) on history
    mod = sm.tsa.ARIMA(
        history,
        order=(2,0,0),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = mod.fit()

    # one‐step forecast
    one = res.get_forecast(steps=1)
    mean = one.predicted_mean.iloc[0]
    ci   = one.conf_int(alpha=0.05).iloc[0]
    se   = one.se_mean.iloc[0]

    # store
    forecasts.append(mean)
    lower95.append(ci[0])
    upper95.append(ci[1])
    ses.append(se)
    actuals.append(ts.loc[period])

# --- 3) assemble results into a DataFrame --------------------
fc = pd.DataFrame({
    'Actual':   actuals,
    'Forecast': forecasts,
    'SE':       ses,
    'Lower95':  lower95,
    'Upper95':  upper95
}, index=fc_periods)

print(fc.head())   # see the first few rolling forecasts

# --- 4) plot Forecast vs Actual with 95% CI ------------------
plt.figure(figsize=(10,6))
x = fc.index.to_timestamp()

plt.plot(  x, fc['Actual'],   label='Actual',  linewidth=1)
plt.plot(  x, fc['Forecast'], label='Forecast', linestyle='--', linewidth=1)
plt.fill_between(x, fc['Lower95'], fc['Upper95'], alpha=0.3, label='95% CI')

plt.title('Rolling One-Step Forecast vs Actuals\n(AR(3) re-estimated each quarter)')
plt.xlabel('Quarter')
plt.ylabel('Initial Claims')
plt.legend()
plt.tight_layout()
plt.show()

############################################################################################################################################################################
# UNBIASNESS OF THE FORECAST 

# compute forecast errors
fc['Error'] = fc['Actual'] - fc['Forecast']

# plot the error time‐series
plt.figure(figsize=(10,6))
x = fc.index.to_timestamp()

plt.plot(x, fc['Error'], label='Forecast Error', linewidth=1)
plt.axhline(0, linestyle='--', linewidth=1)
plt.title('Rolling One-Step Forecast Errors\n(AR(3) re-estimated each quarter)')
plt.xlabel('Quarter')
plt.ylabel('Error (Actual minus Forecast)')
plt.legend()
plt.tight_layout()
plt.show()


errors = fc['Error'].dropna()

# 1) compute stats
mean   = errors.mean()
median = errors.median()
minimum= errors.min()
maximum= errors.max()
std    = errors.std()
skew   = stats.skew(errors)
# fisher=False gives the “raw” kurtosis (not excess)
kurt   = stats.kurtosis(errors, fisher=False)
jb_stat, jb_p = stats.jarque_bera(errors)

# 2) build a text block
stats_text = (
    f"Mean     {mean:8.3f}\n"
    f"Median   {median:8.3f}\n"
    f"Maximum  {maximum:8.3f}\n"
    f"Minimum  {minimum:8.3f}\n"
    f"Std. Dev {std:8.3f}\n"
    f"Skewness {skew:8.3f}\n"
    f"Kurtosis {kurt:8.3f}\n\n"
    f"Jarque–Bera {jb_stat:6.2f}\n"
    f"Prob      {jb_p:.6f}"
)

# 3) plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(errors, bins=20, alpha=0.7, edgecolor='black')
ax.set_title('Histogram of One‐Step Forecast Errors')
ax.set_xlabel('Error (Actual − Forecast)')
ax.set_ylabel('Frequency')

# place the stats box in the upper right
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
ax.text(0.95, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props)

plt.tight_layout()
plt.show()