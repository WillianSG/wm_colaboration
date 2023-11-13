import ast
import os
from collections import defaultdict
from datetime import datetime
from functools import partial
from statistics import mean

import pandas as pd
import numpy as np
from random import randrange

import pathos
import tqdm_pathos
from matplotlib import pyplot as plt

from helper_functions.recurrent_competitive_network import run_rcn


def run_rcn_and_compute_frequency(params, num_attractors=2, num_cues=2):
    # print('-----------------------------------')
    # print('Chosen parameters:')
    # print(f'\tScore: {params["score"]}')
    # print(f'\tAccuracy: {params["accuracy"]}')
    # print(f'\tRecall: {params["recall"]}')
    original_recall = params['recall']
    params = ast.literal_eval(params['params'])
    # print(f'\t{params}')

    params['num_attractors'] = num_attractors
    params['num_cues'] = num_cues
    params['recall_time'] = 4

    rcn, returned_params = run_rcn(params, show_plot=False,
                                   progressbar=False, low_memory=True,
                                   attractor_conflict_resolution="3", return_complete_stats=True)

    # print('\tPS frequency')
    ps_frequencies = {}
    for k, v in returned_params['ps_counts'].items():
        ps_freq = len(v['triggered']) / v['cued_time']
        ps_frequencies[k] = ps_freq
        # print(f"\t\t{k}: {ps_freq} Hz")

    def get_spikes_frequency(spk_mon_ts):
        spikes_cues = defaultdict(list)
        spikes_recall = defaultdict(list)
        for i in returned_params['attractors_cueing_order']:
            spikes_cues[i[0]].append(
                len(np.argwhere((np.array(spk_mon_ts) >= i[3][2][0]) & (np.array(spk_mon_ts) <= i[3][2][1]))) /
                (i[3][2][1] - i[3][2][0]))
            spikes_recall[i[0]].append(
                len(np.argwhere((np.array(spk_mon_ts) > i[3][2][1]) & (np.array(spk_mon_ts) <= i[3][2][1] + i[2]))) / i[
                    2])
        # print('\t\tCued frequency')
        for k, v in spikes_cues.items():
            spikes_cues[k] = np.mean(v)
            # print(f"\t\t\t{k}: {spikes_cues[k]} Hz")
        # print('\t\tRecall frequency')
        for k, v in spikes_recall.items():
            spikes_recall[k] = np.mean(v)
            # print(f"\t\t\t{k}: {spikes_recall[k]} Hz")

        return spikes_cues, spikes_recall

    # print('\tExcitatory spikes frequency')
    spk_mon_ids_e, spk_mon_ts_e = rcn.get_spks_from_pattern_neurons()
    freq_cues_e, freq_recall_e = get_spikes_frequency(spk_mon_ts_e)

    # print('\tInhibitory spikes frequency')
    spk_mon_ids_i, spk_mon_ts_i = rcn.get_I_spks()
    freq_cues_i, freq_recall_i = get_spikes_frequency(spk_mon_ts_i)

    del rcn

    return ps_frequencies, freq_cues_e, freq_recall_e, freq_cues_i, freq_recall_i, returned_params, original_recall


save_folder = f'RESULTS/EFFICIENCY/EFFICIENCY_SAVED_({datetime.now().strftime("%Y-%m-%d_%H-%M-%S")})'
os.makedirs(save_folder)
print("TMP:", save_folder)

num_attractors = 2
cv = 10
num_cues = 100

# load results from RCN Bayesian sweep
df = pd.read_csv("RESULTS/BAYESIAN_OPTIMISATION/R=0.5_BAYESIAN_SAVED_(2023-11-04_13-09-55)/results.csv")

# sweep recalls in [0.1-1] with step 0.1
lower_lim = np.arange(0., 1.0, 0.1)
upper_lim = np.arange(0.1, 1.1, 0.1)
# lower_lim = [0.8, 0.9]
# upper_lim = [0.9, 1.0]

param_grid_pool = []
for r1, r2 in zip(lower_lim, upper_lim):
    queried_param = df.query(f'accuracy == 1 & recall > {r1} & recall <= {r2}').iloc[0]
    param_grid_pool.append(queried_param)

param_grid_pool = [x.copy() for x in param_grid_pool for _ in range(cv)]

results = tqdm_pathos.map(
    partial(
        run_rcn_and_compute_frequency,
        num_attractors=num_attractors,
        num_cues=num_cues
    ),
    param_grid_pool,
    n_cpus=pathos.multiprocessing.cpu_count(),
)


def results_to_dict(col):
    dic = defaultdict(list)
    for p in results:
        dic[p[6]].append(mean(p[col].values()))
    return dic


ps_frequencies_plot_list = results_to_dict(0)
spikes_cues_plot_list_e = results_to_dict(1)
spikes_recall_plot_list_e = results_to_dict(2)
spikes_cues_plot_list_i = results_to_dict(3)
spikes_recall_plot_list_i = results_to_dict(4)

for d in [ps_frequencies_plot_list, spikes_cues_plot_list_e, spikes_recall_plot_list_e, spikes_cues_plot_list_i,
          spikes_recall_plot_list_i]:
    for k, v in d.items():
        d[k] = mean(v)

num_E_neurons = num_attractors * ast.literal_eval(queried_param['params'])['attractor_size']
num_I_neurons = ast.literal_eval(queried_param['params'])['network_size'] // 4

pd.DataFrame({'Recall': spikes_cues_plot_list_e.keys(),
              'Frequency Cue E': np.array(list(spikes_cues_plot_list_e.values())) / num_E_neurons,
              'Frequency Recall E': np.array(list(spikes_recall_plot_list_e.values())) / num_E_neurons,
              'Frequency Cue I': np.array(list(spikes_cues_plot_list_i.values())) / num_I_neurons,
              'Frequency Recall I': np.array(list(spikes_recall_plot_list_i.values())) / num_I_neurons,
              'PS Frequency': np.array(list(ps_frequencies_plot_list.values()))}).to_csv(f'{save_folder}/results.csv')
