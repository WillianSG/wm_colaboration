import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pylab as pl
from sklearn import preprocessing
import matplotlib.ticker as mticker


def dir_walker(root, extension):
    files = []
    for dir in os.listdir(root):
        if os.path.isdir(os.path.join(root, dir)):
            res = dir_walker(os.path.join(root, dir), extension)
            files.extend(res)
        else:
            if dir.endswith(extension):
                files.append(os.path.join(root, dir))

    return files


def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]


files = dir_walker('RESULTS/PARAMETER_ROBUSTNESS/', 'csv')

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
dfs = []
param_names = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)
    df[df.columns[1]] = scaler.fit_transform(df[[df.columns[1]]])
    param_names.append(df.columns[1].replace('_', ' ').title())
dfs.sort(key=lambda df: np.var(df['f1_score'].values), reverse=True)

fig = pl.figure(figsize=(10, 10))
ax = pl.subplot(projection='3d')
for fc, df in enumerate(dfs):
    param_name = df.columns[1]
    y = np.ones_like(df[param_name].values) * fc
    x = df[param_name].values
    z = df['f1_score'].values
    ax.plot(x, y, z)
    xx, zz = np.meshgrid(range(2), range(2))
    yy = np.ones_like(xx) * fc
    ax.plot_surface(xx, yy, zz, color='w', alpha=0.2)

    x_closest_to_mean = np.argmin(np.abs(df[param_name] - np.mean(df[param_name])))
    ax.plot([x[x_closest_to_mean], x[x_closest_to_mean]], [y[0], y[0]], [0, df['f1_score'][x_closest_to_mean]],
            color='r',
            linestyle='--')
    # verts = polygon_under_graph(x, z)
    # poly = plt.Polygon(verts, facecolor='0.9', edgecolor='0.5')

    # ax.add_hrect(y0=0.9, y1=2.6, line_width=0, fillcolor="red", opacity=0.2)
    # for i in range(1, 3):
    #     ax.axvline(x=np.mean(df[param_name]) - i * np.std(df[param_name]), color='k', linestyle='--')
    #     ax.axvline(x=np.mean(df[param_name]) + i * np.std(df[param_name]), color='k', linestyle='--')

# ax.set_xticks([])
ax.set_yticks(range(len(files)))
ax.set_yticklabels(param_names, rotation=-15, va='bottom', ha='right')
ax.set_zlim([0, 1])
ax.set_zlabel('F1 score (z)')
ax.set_box_aspect((3, 5, 1))

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

fig.tight_layout()
fig.suptitle('Parameter value')
pl.show()
