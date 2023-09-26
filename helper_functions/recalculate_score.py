import pandas as pd
import numpy as np

from helper_functions.other import f1_score

# folder = '8ATR_SWEEP_(2023-09-14_20-37-13)'
folder = '4ATR_SWEEP_(2023-08-31_00-39-08)'
df = pd.read_csv(f"../network_dynamics/RCN/intrinsic_plasticity/RESULTS/{folder}/results.csv")

# copy the data
df_min_max_scaled = df.copy()

# apply normalization
df_min_max_scaled['recall'] = (df_min_max_scaled['recall'] - df_min_max_scaled['recall'].min()) / (
        df_min_max_scaled['recall'].max() - df_min_max_scaled['recall'].min())

df_min_max_scaled['score'] = f1_score(df_min_max_scaled['accuracy'], df_min_max_scaled['recall'])
df_min_max_scaled = df_min_max_scaled.fillna(0)

# write the new data
df_min_max_scaled.to_csv(
    f"../network_dynamics/RCN/intrinsic_plasticity/RESULTS/{folder}/results_min_max_scaled.csv",
    index=False)

df_recall_score = df_min_max_scaled.copy()

# Triangle function centred at x=0.5 with base 1: y=max(1-2*abs(x-0.5),0)
optimal_recall = 0.5
df_recall_score['recall'] = df_recall_score['recall'].apply(lambda x: max(1 - 2 * abs(x - optimal_recall), 0))
df_recall_score['score'] = f1_score(df_recall_score['accuracy'], df_recall_score['recall'])
df_recall_score = df_recall_score.fillna(0)

# write the new data
df_recall_score.to_csv(
    f"../network_dynamics/RCN/intrinsic_plasticity/RESULTS/{folder}/results_optimal_recall={optimal_recall}.csv",
    index=False)
