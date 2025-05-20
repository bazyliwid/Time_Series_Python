# import necesary libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_process import arma_acf

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

filtered2 = data_idx.loc['1993Q1':'2009Q4'].reset_index(drop=True)
ts3 = np.log(filtered2['RSFHFS']).diff().dropna()

# The code below estimates AR(p) and MA(q) models for p,q=0,1,2,3 and records the corresponding AIC and BIC values

model_IC_score = pd.DataFrame(columns=['MA_AIC', 'MA_BIC', 'AR_AIC', 'AR_BIC'])

for i in range(4):
    ar_model = sm.tsa.ARIMA(ts3, order=(i,0,0)).fit()
    ma_model = sm.tsa.ARIMA(ts3, order=(0,0,i)).fit()
    model_IC_score.loc[i] = [ma_model.aic, ma_model.bic, ar_model.aic, ar_model.bic]

print(model_IC_score)

ar2_model = sm.tsa.ARIMA(ts3, order=(2,0,0)).fit()
r_1, r_2 = ar2_model.arroots
print(r_1, r_2)
phi1, phi2 = ar2_model.params['ar.L1'], ar2_model.params['ar.L2']
theoretical_acf = arma_acf([1, -phi1, -phi2], [1], lags=20)
print(theoretical_acf)
