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

    fig, (ax_hist, ax_stats) = plt.subplots(1, 2, figsize=(10, 4),
                                            gridspec_kw={'width_ratios':[3, 1]})

    ax_hist.hist(growthRate, bins=30, edgecolor='black')
    ax_hist.set_xlabel('Growth rate (%)')
    ax_hist.set_ylabel('Frequency')

    ax_stats.axis('off')
    text = (
        f'N         {len(growthRate)}\n'
        f'Mean      {mean:8.4f}\n'
        f'Median    {median:8.4f}\n'
        f'Max       {mx:8.4f}\n'
        f'Min       {mn:8.4f}\n'
        f'Std. Dev. {std:8.4f}\n'
        f'Skewness  {skew:8.4f}\n'
        f'Kurtosis  {kurt:8.4f}\n'
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




# results = []

# for group in groups:
#     # Dates to remove
#     dates = growthRate.index[group]
#     # Remove these dates from the series
#     filtered = growthRate.drop(dates)
#     jb_stat, jb_pval, jb_skew, jb_kurt = jarque_bera(filtered)
#     # Format dates as string
#     if len(dates) == 1:
#         date_str = dates[0].strftime('%Y-%m-%d')
#     else:
#         date_str = f"{dates[0].strftime('%Y-%m-%d')}–{dates[-1].strftime('%Y-%m-%d')}"
#     results.append([date_str, jb_stat, jb_pval, jb_skew, jb_kurt])

# # Create DataFrame
# df_outlier_jb = pd.DataFrame(
#     results,
#     columns=[
#         'Removed Date(s)',
#         'JB_stat',
#         'JB_pval',
#         'JB_skew',
#         'JB_kurt'
#     ]
# )

# print(df_outlier_jb)


#     labels.append(f"{start_date}–{end_date}")
    
#     # run JB
#     jb_stat, jb_pval, _, _ = jarque_bera(block)
#     jb_stats.append(jb_stat)
#     jb_pvals.append(jb_pval)

# # assemble into a DataFrame
# df_jb = pd.DataFrame({
#     'JB_stat': jb_stats,
#     'JB_pval': jb_pvals
# }, index=labels)

# # ------ plotting ------
# alpha = 0.05

# # assume df_jb is your DataFrame of results
# #     index = ["YYYY-MM-DD–YYYY-MM-DD", …]
# #     columns = ["JB_stat", "JB_pval"]

# # prepare a list of colours for each tick
# tick_colors = [
#     'red' if pval < alpha else 'black'
#     for pval in df_jb['JB_pval']
# ]

# fig, ax1 = plt.subplots(figsize=(10,5))

# # 1) JB statistic as bars
# ax1.bar(
#     df_jb.index,
#     df_jb['JB_stat'],
#     label='JB statistic',
#     zorder=1
# )
# ax1.set_ylabel('Jarque–Bera statistic')
# ax1.set_xlabel('Period (start–end)')

# # set ticks & labels (without trying to pass a list of colors)
# ax1.set_xticks(range(len(df_jb)))
# ax1.set_xticklabels(df_jb.index, rotation=45, ha='right')

# # now color them individually
# for tick_label, color in zip(ax1.get_xticklabels(), tick_colors):
#     tick_label.set_color(color)

# # 2) JB p-value as red line
# ax2 = ax1.twinx()
# ax2.plot(
#     df_jb.index,
#     df_jb['JB_pval'],
#     marker='o',
#     color='red',
#     label='JB p-value',
#     zorder=2
# )
# ax2.axhline(
#     y=alpha,
#     color='grey',
#     linestyle='--',
#     linewidth=1,
#     label=f'α = {alpha}',
#     zorder=0
# )
# ax2.set_ylabel('JB p-value')

# # 3) legend
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# plt.title(f'Jarque–Bera per Bin (α = {alpha})')
# plt.tight_layout()
# plt.show()
