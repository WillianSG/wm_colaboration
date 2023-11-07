import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib import ticker

df = pd.read_csv('RESULTS/EFFICIENCY_SAVED_(2023-11-07_10-35-01)/results.csv')

sns.set_style('darkgrid', {'axes.grid': False})  # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)  # fontsize of the axes title
plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)  # fontsize of the tick labels
plt.rc('ytick', labelsize=13)  # fontsize of the tick labels
plt.rc('legend', fontsize=13)  # legend fontsize
plt.rc('font', size=13)  # controls default text sizes

fig, axes = plt.subplots(2, 1, figsize=(7, 6))
axes[0].xaxis.set_major_locator(ticker.MultipleLocator(0.1))
axes[0].plot(np.arange(0.1, 1.1, 0.1), df['Frequency Recall E'].values + df['Frequency Recall I'].values, 'o',
             color=sns.color_palette('deep')[0], label='E + I neurons', linewidth=2, linestyle='--')
# axes[0].plot(0.5, df['Frequency Recall E'].values[4] + df['Frequency Recall I'].values[4], color='black', marker='x')
axes[0].axvline(x=0.5, alpha=0.7, color='black', linestyle='--', linewidth=1)
ax2 = axes[0].twinx()
ax2.plot(np.arange(0.1, 1.1, 0.1), df['PS Frequency'].values, 'o-', color=sns.color_palette('deep')[1], label='PSs',
         linewidth=2, linestyle='--')
# ax2.plot(0.5, df['PS Frequency'].values[4], color='black', marker='x')
axes[0].set_ylabel('E+I frequency (Hz)', color=sns.color_palette('deep')[0])
ax2.set_ylabel('PS frequency (Hz)', color=sns.color_palette('deep')[1])
axes[0].set_xticklabels([])

axes[1].xaxis.set_major_locator(ticker.MultipleLocator(0.1))
axes[1].plot(np.arange(0.1, 1.1, 0.1), df['Frequency Cue E'].values + df['Frequency Cue I'].values, 'o',
             label='Cue period',
             color=sns.color_palette('deep')[2], linewidth=2, linestyle='--')
# axes[1].plot(0.5, df['Frequency Cue E'].values[4] + df['Frequency Cue I'].values[4], color='black', marker='x')
axes[1].plot(np.arange(0.1, 1.1, 0.1), df['Frequency Recall E'].values + df['Frequency Recall I'].values, 'o',
             label='Recall period', color=sns.color_palette('deep')[3], linewidth=2, linestyle='--')
# axes[1].plot(0.5, df['Frequency Recall E'].values[4] + df['Frequency Recall I'].values[4], color='black', marker='x')
axes[1].axvline(x=0.5, alpha=0.7, color='black', linestyle='--', linewidth=1)
axes[1].set_xlabel(r'Recall ($R$)')
axes[1].set_ylabel('E+I frequency (Hz)')
axes[1].legend()
fig.tight_layout()
fig.show()
fig.savefig('RESULTS/EFFICIENCY/efficiency.pdf')

print(df['Frequency Cue E'].mean())
print(df['Frequency Recall E'].mean())
print('Average improvement', df['Frequency Cue E'].mean() / df['Frequency Recall E'].mean())

print(df['Frequency Cue E'][4])
print(df['Frequency Recall E'][4])
print('R=0.5 improvement', df['Frequency Cue E'][4] / df['Frequency Recall E'][4])
