'''
https://stackoverflow.com/questions/67585809/how-to-map-the-results-of-principal-component-analysis-back-to-the-actual-featur
'''

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle, sys, os
import numpy as np
import pickle

if sys.platform in ['linux', 'win32']:

    root = os.path.dirname(os.path.abspath(os.path.join(__file__ , '../../../')))

sys.path.append(os.path.join(root, 'helper_functions'))

path = 'D://A_PhD//GitHub//wm_colaboration//results//sFSA_surface_gauss_search'

files = os.listdir(path)

gauss_search_data = {}

for file in files:

    with open(
        f'{path}//{file}',
        'rb') as f:(
        pickle_dat) = pickle.load(f)

    for param, val in pickle_dat.items():

        if param not in gauss_search_data:
            gauss_search_data[param] = []

        gauss_search_data[param].append(val)

data = pd.DataFrame.from_dict(gauss_search_data)

for col in data.columns:
    data.drop(data[data[col] < 0].index, inplace = True)

# Separate the parameters and CR columns
parameters = data.drop('CR', axis=1)
cr = data['CR']

# Store the parameter column names
parameter_names = parameters.columns.tolist()
print(parameter_names)

# Standardize the parameters
scaler = StandardScaler()
parameters_scaled = scaler.fit_transform(parameters)

# Perform PCA
pca = PCA()
principal_components = pca.fit_transform(parameters_scaled)

# Analyze explained variance
explained_variance_ratio = pca.explained_variance_ratio_
total_variance_explained = sum(explained_variance_ratio)

# Print explained variance ratio for each principal component
for i, explained_var in enumerate(explained_variance_ratio):
    print(f"Explained Variance of PC{i + 1}: {explained_var:.4f}")

print(f"Total Variance Explained: {total_variance_explained:.4f}")

# Analyze component loadings
component_loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'])  # Add column names accordingly

# Sort component loadings by absolute values for each principal component
sorted_loadings = component_loadings.abs().sort_values(by='PC1', ascending=False)

# Print the top influential parameters for the first principal component (PC1)
top_parameters_pc1 = sorted_loadings['PC1'].head(5)  # Adjust the number of parameters to display
print("Top Parameters for PC1:")
print(top_parameters_pc1)

# Plot explained variance ratio
import matplotlib.pyplot as plt

plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio per Principal Component')
plt.show()
plt.close()

# Plot the loadings as a bar plot
loadings = pd.DataFrame(pca.components_, columns=['PC1', 'PC2', 'PC3'], index=parameter_names)
loadings.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Parameters')
plt.ylabel('Loadings')
plt.title('PCA Loadings for Parameters')
plt.legend(loc = 'best', framealpha = 0.0)
plt.xticks(rotation=45)
#plt.ylim(-0.8, 0.8)
plt.grid(axis='y', color = 'gray', ls = '--', lw = 1.0, alpha = 0.5)
plt.show()

print(loadings)

PCs = ['PC1', 'PC2', 'PC3']
data = {}

for pc in PCs:
    data[pc] = loadings[pc]

df = pd.DataFrame(data, index=parameter_names)

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(df.columns))

width = 0.1

for i, row in enumerate(df.iterrows()):
    name, values = row
    offset = i * width
    ax.bar(x + offset, values, width, label=name)

ax.set_xticks(x)
ax.set_xticklabels(df.columns)
ax.set_ylabel('loadings')

#plt.ylim(-0.8, 0.8)
plt.grid(axis='y', color = 'gray', ls = '--', lw = 1.0, alpha = 0.5)

plt.legend(loc = 'best', framealpha = 0.0)
plt.show()