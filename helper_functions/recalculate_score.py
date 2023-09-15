import pandas as pd

from helper_functions.other import f1_score

folder = '8ATR_SWEEP_(2023-09-14_20-37-13)'
df = pd.read_csv(f"../network_dynamics/RCN/intrinsic_plasticity/RESULTS/{folder}/results.csv")

# copy the data
df_min_max_scaled = df.copy()

# apply normalization techniques
df_min_max_scaled['recall'] = (df_min_max_scaled['recall'] - df_min_max_scaled['recall'].min()) / (
        df_min_max_scaled['recall'].max() - df_min_max_scaled['recall'].min())
df_min_max_scaled['score'] = f1_score(df_min_max_scaled['accuracy'], df_min_max_scaled['recall'])

# write the new data
df_min_max_scaled.to_csv(
    f"../network_dynamics/RCN/intrinsic_plasticity/RESULTS/{folder}/results_min_max_scaled.csv",
    index=False)
