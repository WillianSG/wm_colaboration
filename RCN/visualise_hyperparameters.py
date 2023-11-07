import numpy as np
import pandas as pd
import time

import plotly.io as plt_io
from chart_studio import plotly as py
import plotly.graph_objects as pgo
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

df = pd.read_csv('RESULTS/BAYESIAN_OPTIMISATION/R=0.5_BAYESIAN_SAVED_(2023-11-04_13-09-55)/results.csv')
df = df.query('score > 0.99')

params = pd.DataFrame.from_dict(df['params'].apply(lambda x: eval(x)).tolist())[
    [
        'background_activity',
        'i_frequency',
        'i_e_weight',
        'e_i_weight',
        'e_e_max_weight',
    ]]
param_latex = {'background_activity': '$E_{FREQ}$',
               'e_e_max_weight': '$W_{EE}$',
               'e_i_weight': '$W_{EI}$',
               'i_e_weight': '$W_{IE}$',
               'i_frequency': '$I_{FREQ}$'}

## Standardizing the data
x = StandardScaler().fit_transform(params)
features = params.columns

fig = px.scatter_matrix(
    params,
    dimensions=features,
    labels={col: param_latex[col] for col in features},
)
fig.update_traces(diagonal_visible=False)
fig.show()
fig.write_image('RESULTS/VISUALISE_HYPER/params_scatter_matrix.pdf', width=1000, height=1000)

pca = PCA()
components = pca.fit_transform(x)
data4 = [
    pgo.Scatter(
        y=np.cumsum(pca.explained_variance_ratio_),
    )
]
fig = pgo.Figure(data=data4)
fig.show()
fig.write_image('RESULTS/VISUALISE_HYPER/pca_cumulative_variance.pdf')

pca_components = 3
pca = PCA(n_components=pca_components)
components = pca.fit_transform(x)
df_components = pd.DataFrame(components, index=df.index)
fig = px.scatter_matrix(
    components,
    dimensions=range(pca_components),
    labels={col: col.replace('_', ' ').title() for col in features},
)
fig.update_traces(diagonal_visible=False)
fig.show()
fig.write_image('RESULTS/VISUALISE_HYPER/pca_scatter_matrix.pdf')


def cluster(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(components)
    Z = kmeans.predict(components)
    return kmeans, Z


max_clusters = len(df)
inertias = np.zeros(max_clusters)

for i in range(1, max_clusters):
    kmeans, Z = cluster(i)
    inertias[i] = kmeans.inertia_

data6 = pgo.Scatter(
    x=list(range(1, max_clusters)),
    y=inertias[1:]
)

layout6 = pgo.Layout(
    title='Investigate k-means clustering',
    xaxis=pgo.layout.XAxis(title='Number of clusters',
                           range=[0, max_clusters]),
    yaxis=pgo.layout.YAxis(title='Inertia')
)

fig = pgo.Figure(data=data6, layout=layout6)
fig.show()

n_clusters = 5
model, Z = cluster(n_clusters)

trace0 = pgo.Scatter(x=df_components[0],
                     y=df_components[1],
                     text=df.index,
                     name='',
                     mode='markers',
                     marker=pgo.scatter.Marker(size=df['score'],
                                               sizemode='diameter',
                                               sizeref=df['score'].max() / 50,
                                               opacity=0.5,
                                               color=Z),
                     showlegend=False
                     )

trace1 = pgo.Scatter(x=model.cluster_centers_[:, 0],
                     y=model.cluster_centers_[:, 1],
                     name='',
                     mode='markers',
                     marker=pgo.scatter.Marker(symbol='x',
                                               size=12,
                                               color=list(range(n_clusters))),
                     showlegend=False
                     )
data7 = [trace0, trace1]
layout7 = pgo.Layout(title='Baltimore Vital Signs (PCA)',
                     xaxis=pgo.layout.XAxis(showgrid=False,
                                            zeroline=False,
                                            showticklabels=False),
                     yaxis=pgo.layout.YAxis(showgrid=False,
                                            zeroline=False,
                                            showticklabels=False),
                     hovermode='closest'
                     )
layout7['title'] = f'PCA and k-means clustering with {n_clusters} clusters'
fig = pgo.Figure(data=data7, layout=layout7)
fig.show()
fig.write_image('RESULTS/VISUALISE_HYPER/pca_kmeans.pdf')
