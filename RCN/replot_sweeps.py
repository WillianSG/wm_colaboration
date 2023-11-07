import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn import preprocessing
import matplotlib.ticker as mticker
import matplotlib.colors as cm

from helper_functions.other import dir_walker


def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]


folder = 'RESULTS/PARAMETER_ROBUSTNESS_R_0-5'
files = dir_walker(folder, 'results.csv')

dfs = []
means = {}
for f in files:
    df = pd.read_csv(f)

    # if df.columns[1] == 'e_e_max_weight':
    #     continue
    df.sort_values(by=df.columns[1], inplace=True)
    dfs.append(df)
    for col in df.columns[2:6]:
        if col not in means:
            means[col] = df[col].values[0]
dfs.sort(key=lambda df: np.var(df['f1_score'].values), reverse=True)

fig = pl.figure(figsize=(10, 10))
ax = pl.subplot(projection='3d', computed_zorder=False)
colors = [cm.to_hex(plt.cm.Set1(i)) for i in range(len(dfs))]
for fc, df in enumerate(dfs):
    param_name = df.columns[1]

    y = np.ones_like(df[param_name].values) * fc
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    x = scaler.fit_transform(df[param_name].values.reshape(-1, 1)).squeeze()
    z = df['f1_score'].values

    ax.plot(x, y, z, color=colors[fc], linewidth=2, zorder=len(dfs) - fc)


    def closest_value(value):
        return np.argmin(np.abs(df[param_name] - value))


    x_mean = df.index[np.isclose(df[param_name], means[param_name])].tolist()[0]
    sigma_bounds = [(closest_value(means[param_name] - s * np.std(df[param_name])),
                     closest_value(means[param_name] + s * np.std(df[param_name]))) for s in range(1, 4)]

    if fc == len(dfs) - 1:
        ax.text(x[x_mean], y[0], 1, r'$\mu$', 'x', zorder=1000, ha='center', va='center',
                bbox=dict(boxstyle='round', fc='w', ec='k', alpha=0.5))
        ax.text(x[sigma_bounds[0][0]], y[0], 1, r'$\sigma$', 'x', zorder=1000, ha='center', va='center',
                bbox=dict(boxstyle='round', fc='w', ec='k', alpha=0.5))
        ax.text(x[sigma_bounds[1][0]], y[0], 1, r'$2\sigma$', 'x', zorder=1000, ha='center', va='center',
                bbox=dict(boxstyle='round', fc='w', ec='k', alpha=0.5))
        ax.text(x[sigma_bounds[2][0]], y[0], 1, r'$3\sigma$', 'x', zorder=1000, ha='center', va='center',
                bbox=dict(boxstyle='round', fc='w', ec='k', alpha=0.5))

    verts = [polygon_under_graph(x[s[0]:s[1]], z[s[0]:s[1]]) for s in sigma_bounds]
    ax.add_collection3d(PolyCollection(verts, facecolors=colors[fc], alpha=.4, zorder=len(dfs) - fc),
                        zs=[y[0]], zdir='y')
    ax.plot([x[x_mean], x[x_mean]], [y[0], y[0]], [0, df['f1_score'][x_mean]],
            color=colors[fc],
            linestyle='--',
            zorder=len(dfs) - fc)

ax.set_xticks([])
# ax.set_ylabel('Parameter name (Increasing variance)')
ax.set_yticks(range(len(dfs)))
ax.set_zlim([0, 1])
ax.set_zlabel('F1-score')
ax.set_box_aspect((3, 5, 1))
ax.tick_params(axis='y', width=10, labelsize=10, pad=0)

param_latex = {'background_activity': '$E_{FREQ}$',
               'e_e_max_weight': '$W_{EE}$',
               'e_i_weight': '$W_{EI}$',
               'i_e_weight': '$W_{IE}$',
               'i_frequency': '$I_{FREQ}$'}
param_names = []
for df in dfs:
    param_names.append(df.columns[1])

ax.set_yticklabels([f'{p.replace("_", " ").title()} ({param_latex[p]})' for p in param_names], rotation=-15,
                   va='center_baseline', ha='left')
ax.grid(True)

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

fig.tight_layout()
# ax.set_title('Parameter value')
pl.show()
fig.savefig(f'{folder}/robustness.pdf', bbox_inches='tight')
