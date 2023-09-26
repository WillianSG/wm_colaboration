import ast

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from boruta import BorutaPy
from sklearn.linear_model import LogisticRegression

# folder = '8ATR_SWEEP_(2023-09-14_20-37-13)'
folder = '4ATR_SWEEP_(2023-08-31_00-39-08)'
df = pd.read_csv(
    f"../network_dynamics/RCN/intrinsic_plasticity/RESULTS/{folder}/results_optimal_recall=0.5.csv")

df_params = pd.DataFrame.from_records([ast.literal_eval(p) for p in df['params']])
X = df_params.values
y = df['score'].values

# load X and y
# NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
# X = pd.read_csv('test_X.csv', index_col=0).values
# y = pd.read_csv('test_y.csv', header=None, index_col=0).values
y = y.ravel()

# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
rf = RandomForestRegressor(n_jobs=-1, max_depth=5)

# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

# find all relevant features
feat_selector.fit(X, y)

# check selected features
# feat_selector.support_
#
# # check ranking of features
# feat_selector.ranking_

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X)

# make it into a dataframe again
X_filtered = pd.DataFrame(X_filtered, columns=df_params.columns[feat_selector.support_])

print('Original features:\n', list(df_params.columns))
print('Selected features:\n', list(df_params.columns[feat_selector.support_]))

# save the dataframe
X_filtered.to_csv(f"../network_dynamics/RCN/intrinsic_plasticity/RESULTS/{folder}/X_feature_selection.csv",
                  index=False)
