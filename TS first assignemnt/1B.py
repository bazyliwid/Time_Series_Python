import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm 
from statsmodels.stats.stattools import jarque_bera
import numpy as np
import math
from itertools import groupby
from operator import itemgetter


data = pd.read_excel(r'C:\Users\leoro\OneDrive\Desktop\EUR\YEAR 3\BLOCK 5\Time Series\Assignment 1\data.xlsx',header=0,parse_dates=["QUARTER"],index_col="QUARTER")

# Extract the data for the quaterly growth rate of the seasonally adjusted series
seasonadjustedY = data['RSFHFS']

loggedseasonadjustedY = np.log(seasonadjustedY)

growthRate = loggedseasonadjustedY.diff() * 100

growthRate = growthRate.dropna()

print(growthRate)	

# Function that tests for normality for inputted time series
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
test_normality(growthRate)

# To identify the periods most influential to violations of normality, identify outliers where growth rate < -threshold or > threshold
threshold = 4
outlier_mask = (growthRate < -threshold) | (growthRate > threshold)
outlier_indices = growthRate.index[outlier_mask]

filtered = growthRate.drop(outlier_indices)

print("Outlier indices:", outlier_indices)

# Test for normality after removing outliers	
test_normality(filtered)