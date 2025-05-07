# libraries import 
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# thats for me 
# to activate: conda activate /opt/anaconda3/envs/test
# to run: python week_1.py

############################################################################################################################################################################

# Load the data (paste ur own path)
data_1 = pd.read_excel('/Users/bazyliwidawski/Documents/Block 5, Year 3/Time Series Analysis/Python replication/USUNINSCE.xlsx', parse_dates=['Quarter'], index_col='Quarter')
data_2 = pd.read_excel('/Users/bazyliwidawski/Documents/Block 5, Year 3/Time Series Analysis/Python replication/USCC.xlsx', parse_dates=['Quarter'], index_col='Quarter')
data_3 = pd.read_excel('/Users/bazyliwidawski/Documents/Block 5, Year 3/Time Series Analysis/Python replication/LSAGDP.xlsx', parse_dates=['Quarter'], index_col='Quarter')

############################################################################################################################################################################

# 1) Plot the time series , filtering too easy to be included
plt.figure(figsize=(10, 6))
plt.plot(data_1.index, data_1['US initial claims for unemployment insurance'], label='US Initial Claims for Unemployment Insurance',color='purple')
plt.title('US Initial Claims for Unemployment Insurance')
plt.xlabel('Date')
plt.ylabel('Claims')
plt.legend()
plt.show()

############################################################################################################################################################################

# 3) Function to resample data to higher (here, yearly) frequency with specified aggregation
def plot_resampled_yearly(data, column_name, agg='mean', color='blue', title=None):
    # Resample to yearly frequency using specified aggregation method
    if agg == 'mean':
        data_yearly = data[column_name].resample('Y').mean()
    elif agg == 'sum':
        data_yearly = data[column_name].resample('Y').sum()
    elif agg == 'max':
        data_yearly = data[column_name].resample('Y').max()
    elif agg == 'min':
        data_yearly = data[column_name].resample('Y').min()
    else:
        raise ValueError("Aggregation method not supported. Use 'mean', 'sum', 'max', or 'min'.")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(data_yearly.index, data_yearly, label=f'{column_name} (Yearly - {agg})', color=color)
    plt.title(title or f'{column_name} (Yearly - {agg})')
    plt.xlabel('Year')
    plt.ylabel('Value')
    # plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_resampled_yearly(
    data=data_1,
    column_name='US initial claims for unemployment insurance',
    agg='mean',
    color='purple',
    title='US Initial Claims for Unemployment Insurance (Yearly Mean)'
)

############################################################################################################################################################################

# 4) Test for the presence of a trend, visualize the results

# Extract the series
series = data_1['US initial claims for unemployment insurance'].dropna()

# Create time trend: numerical index
time = range(len(series))
X = sm.add_constant(time)  # Adds intercept

# Regress on linear trend
model = sm.OLS(series.values, X)
results = model.fit()

# Fitted values
trend_line = results.fittedvalues

# Plot original series and trend
plt.figure(figsize=(10, 6))
plt.plot(series.index, series.values, label='Original Series', color='purple')
plt.plot(series.index, trend_line, label='Linear Trend', color='red', linestyle='--')
plt.title('US Initial Claims for Unemployment Insurance with Linear Trend')
plt.xlabel('Date')
plt.ylabel('Claims')
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.show()

# Print the regression result
print(results.summary())

############################################################################################################################################################################

# 5) Plot the data by season on the same graph 
data = data = pd.read_excel('/Users/bazyliwidawski/Documents/Block 5, Year 3/Time Series Analysis/Python replication/USUNINSCE.xlsx')
data[['Year', 'QuarterNum']] = data['Quarter'].str.extract(r'(\d{4})Q([1-4])')
data['Year'] = data['Year'].astype(int)
data['QuarterNum'] = data['QuarterNum'].astype(int)

# Pivot the data: rows = Year, columns = Q1–Q4
pivoted = data.pivot(index='Year', columns='QuarterNum', values='US initial claims for unemployment insurance')
pivoted.columns = ['Q1', 'Q2', 'Q3', 'Q4']

# Plot each quarter
colors = {'Q1': 'blue', 'Q2': 'red', 'Q3': 'green', 'Q4': 'black'}
plt.figure(figsize=(12, 6))
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    plt.plot(pivoted.index, pivoted[q], label=q, color=colors[q])
plt.title('US Initial Claims by Quarter')
plt.xlabel('Year')
plt.ylabel('Claims')
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.show()

############################################################################################################################################################################

# 6) Regression for teh seasonal effects 
data = pd.read_excel(
    '/Users/bazyliwidawski/Documents/Block 5, Year 3/Time Series Analysis/Python replication/USUNINSCE.xlsx'
)

# Extract year and quarter number from 'Quarter' string like '1990Q1'
data[['Year', 'QuarterNum']] = data['Quarter'].str.extract(r'(\d{4})Q([1-4])')
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
data['QuarterNum'] = pd.to_numeric(data['QuarterNum'], errors='coerce')

# Drop rows with missing values in key columns
data = data.dropna(subset=['Year', 'QuarterNum', 'US initial claims for unemployment insurance'])

# Manually create dummy variables (Q1 is baseline)
data['Q1'] = (data['QuarterNum'] == 1).astype(int)
data['Q2'] = (data['QuarterNum'] == 2).astype(int)
data['Q3'] = (data['QuarterNum'] == 3).astype(int)
data['Q4'] = (data['QuarterNum'] == 4).astype(int)

# Use all four dummies, but drop intercept
X = data[['Q1', 'Q2', 'Q3', 'Q4']]
y = data['US initial claims for unemployment insurance']

# Fit model with no intercept
model = sm.OLS(y, X).fit()

print(model.summary())

# F-test for the coefficients 
wald_test = model.f_test(["Q1 - Q3 = 0", "Q1 - Q4 = 0",'Q1 - Q2 = 0'])
print(wald_test)

############################################################################################################################################################################

# 7) Actual vs Predicted and Residuals => it works just looks like shit due to lack of seasonality of data 
# Add fitted and residuals to dataframe
data['Fitted'] = model.fittedvalues
data['Residual'] = model.resid
data['Actual'] = y

# Create time labels from original Quarter string
data['TimeLabel'] = data['Year'].astype(str) + 'Q' + data['QuarterNum'].astype(str)

# Plot: stacked subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# --- Top: Actual vs. Fitted ---
ax1.plot(data.index, data['Actual'], label='Actual', color='orange', linewidth=1)
ax1.plot(data.index, data['Fitted'], label='Fitted', color='green', linewidth=1)
ax1.set_ylabel('Claims')
ax1.set_title('Actual vs Fitted')
ax1.legend()
# ax1.grid(True)

# --- Bottom: Residuals ---
ax2.plot(data.index, data['Residual'], label='Residual', color='steelblue', linewidth=0.9)
ax2.axhline(0, color='gray', linestyle='--', linewidth=0.7)
ax2.set_ylabel('Residuals')
ax2.set_xlabel('Observation (Quarterly Index)')
ax2.set_title('Residuals')
ax2.legend()
# ax2.grid(True)

plt.tight_layout()
plt.show()

############################################################################################################################################################################

# 8) Regression for the recession effects and its impact on volatility
data = pd.read_excel('/Users/bazyliwidawski/Documents/Block 5, Year 3/Time Series Analysis/Python replication/USUNINSCE.xlsx')

# Ensure numeric dtype
data['DREC'] = data['DREC'].astype(int)
data['NOT_DREC'] = 1 - data['DREC']  # This will be the dummy for non-recession

# Define X and y
X = data[['DREC', 'NOT_DREC']]
y = data['US initial claims for unemployment insurance']

# === Step 1: Run the OLS original  ===
model = sm.OLS(y, X).fit()

print(model.summary())

# === Step 2: Get residuals and square them ===
data['resid'] = model.resid
data['resid_sq'] = data['resid'] ** 2

# === Step 3: Regress squared residuals on DREC and NOT_DREC ===
X_var = sm.add_constant(data[['DREC', 'NOT_DREC']])  # Add constant for flexibility
y_var = data['resid_sq']
model_var = sm.OLS(y_var, X_var).fit()

############################################################################################################################################################################

# Load your dataset
data = pd.read_excel('/Users/bazyliwidawski/Documents/Block 5, Year 3/Time Series Analysis/Python replication/USUNINSCE.xlsx')

# Ensure sorting by time
data = data.sort_values(by='Quarter').reset_index(drop=True)

# Calculate first difference (change in explanatory variable)
data['Claims_diff'] = data['US initial claims for unemployment insurance'].diff()

# Calculate 20-quarter rolling standard deviation of the changes
data['Rolling_Std_20Q'] = data['Claims_diff'].rolling(window=20).std()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(data['Quarter'], data['Rolling_Std_20Q'], label='20-Quarter Moving Std Dev of ΔClaims', color='purple')
plt.axhline(data['Rolling_Std_20Q'].mean(), linestyle='--', color='red', label='Mean Volatility')
plt.title('20-Quarter Moving Standard Deviation of Changes in Unemployment Claims')
plt.xlabel('Quarter')
plt.ylabel('Standard Deviation')

# Set x-ticks every 10 observations
tick_positions = np.arange(0, len(data), 10)
plt.xticks(tick_positions, data['Quarter'].iloc[tick_positions], rotation=45)

# plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
