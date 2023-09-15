import ast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from helper_functions.recurrent_competitive_network import run_rcn

# get parameters maximising recall
df = pd.read_csv("RESULTS/4ATR_SWEEP_(2023-08-31_00-39-08)/results.csv")
params = ast.literal_eval(df.iloc[df['recall'].idxmax()]['params'])
params['num_cues'] = 1

results = run_rcn(
    params,
    show_plot=True,
    low_memory=False,
    progressbar=True,
    attractor_conflict_resolution="3"
)

# compute reactivation frequency
print("Reactivation frequency: ")
for k, v in results['ps_counts'].items():
    print(k, len(v['triggered']) / v['cued_time'])
