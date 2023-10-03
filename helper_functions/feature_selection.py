import ast

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from boruta import BorutaPy

folder = '4ATR_SWEEP_(2023-08-31_00-39-08)'
folder = '8ATR_SWEEP_(2023-09-14_20-37-13)'
df = pd.read_csv(
    f"../network_dynamics/RCN/intrinsic_plasticity/RESULTS/{folder}/results_optimal_recall=0.5.csv")

# Select only accuracy=1 and remove zero-variance features
df_filtered = df.query(f'accuracy == 1')
df_filtered = df
df_params = pd.DataFrame.from_records([ast.literal_eval(p) for p in df_filtered['params']])

# load X and y
# NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
variance_selector = VarianceThreshold()
X = variance_selector.fit_transform(df_params)
y = df_filtered['score'].values

# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
rf = RandomForestRegressor(n_jobs=-1, max_depth=5)

# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', max_iter=100, verbose=2, random_state=1)

# find all relevant features
feat_selector.fit(X, y)

# check selected features
# feat_selector.support_
#
# # check ranking of features
# feat_selector.ranking_

# call transform() on X to filter it down to selected features
X_selected = feat_selector.transform(X)

print('Original features and assigned importance:\n',
      [f'{a} ({b})' for a, b in zip(list(df_params.columns[variance_selector.get_support()]), feat_selector.ranking_)])
print('Selected features:\n', list(df_params.columns[variance_selector.get_support()][feat_selector.support_]))
print('Weak features:\n', list(df_params.columns[variance_selector.get_support()][feat_selector.support_weak_]))
