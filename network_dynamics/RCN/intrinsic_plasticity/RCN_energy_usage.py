import ast
from collections import defaultdict
from statistics import mean

import pandas as pd
import numpy as np
from random import randrange

from matplotlib import pyplot as plt

from helper_functions.recurrent_competitive_network import run_rcn


def run_rcn_and_compute_frequency(params, num_attractors=2, num_cues=2, energy_metric='biological'):
    if energy_metric == 'biological':
        energy_per_spike = 6.67e-12

    print('-----------------------------------')
    print('Chosen parameters:')
    print(f'\tScore: {params["score"]}')
    print(f'\tAccuracy: {params["accuracy"]}')
    print(f'\tRecall: {params["recall"]}')
    params = ast.literal_eval(params['params'])
    print(f'\t{params}')

    params['num_attractors'] = num_attractors
    params['num_cues'] = num_cues
    params['recall_time'] = 4

    rcn, returned_params = run_rcn(params, show_plot=True, progressbar=True, low_memory=False,
                                   attractor_conflict_resolution="3", return_complete_stats=True,
                                   seed_init=seed)

    print('\tPS frequency')
    ps_frequencies = {}
    for k, v in returned_params['ps_counts'].items():
        ps_freq = len(v['triggered']) / v['cued_time']
        ps_frequencies[k] = ps_freq
        print(f"\t\t{k}: {ps_freq} Hz")

    print('\tSpikes frequency')
    spk_mon_ids, spk_mon_ts = rcn.get_spks_from_pattern_neurons()

    spikes_cues = defaultdict(list)
    spikes_recall = defaultdict(list)
    for i in returned_params['attractors_cueing_order']:
        spikes_cues[i[0]].append(
            len(np.argwhere((np.array(spk_mon_ts) >= i[3][2][0]) & (np.array(spk_mon_ts) <= i[3][2][1]))) /
            (i[3][2][1] - i[3][2][0]))
        spikes_recall[i[0]].append(
            len(np.argwhere((np.array(spk_mon_ts) > i[3][2][1]) & (np.array(spk_mon_ts) <= i[3][2][1] + i[2]))) / i[2])
    print('\t\tCued frequency')
    for k, v in spikes_cues.items():
        spikes_cues[k] = np.mean(v)
        print(f"\t\t\t{k}: {spikes_cues[k]} Hz")
    print('\t\tRecall frequency')
    for k, v in spikes_recall.items():
        spikes_recall[k] = np.mean(v)
        print(f"\t\t\t{k}: {spikes_recall[k]} Hz")

    # energy consumption
    if energy_metric:
        total_cued_time = 0
        for i in returned_params['attractors_cueing_order']:
            total_cued_time += i[3][2][1] - i[3][2][0]
        total_recalled_time = 0
        for i in returned_params['attractors_cueing_order']:
            total_recalled_time += i[2]

        total_spikes_cues = 0
        for i in spikes_cues.values():
            total_spikes_cues += np.sum(i)
        total_spikes_recall = 0
        for i in spikes_recall.values():
            total_spikes_recall += np.sum(i)

        energy_cues = energy_per_spike * total_spikes_cues / total_cued_time
        energy_recalls = energy_per_spike * total_spikes_recall / total_recalled_time

        print('\tEnergy consumption')
        print(f'\t\tAverage energy consumption per spike ({energy_metric} neuron): ', energy_per_spike, 'J')
        print('\t\tAverage energy consumption per second during cues: ', energy_cues, 'J')
        print('\t\tAverage energy consumption per second during recall: ', energy_recalls, 'J')
        print(
            f'\t\tEnergy improvement per second during recall: {energy_cues - energy_recalls} J ({energy_cues / energy_recalls:.2f}-fold decrease)')

    return ps_frequencies, spikes_cues, spikes_recall, energy_cues, energy_recalls, returned_params


num_attractors = 2
num_cues = 100
seed = randrange(1000000)
# load results from RCN Bayesian sweep
df = pd.read_csv("RESULTS/4ATR_SWEEP_(2023-08-31_00-39-08)/results_min_max_scaled.csv")
# original parameters
# run_rcn_and_compute_frequency(pd.Series({'score': -1, 'accuracy': -1, 'recall': -1,
#                                          'params': "{'attractor_size': 64, 'background_activity': 15, 'cue_length': 1, 'cue_percentage': 100, 'e_e_max_weight': 10, 'e_i_weight': 3, 'i_e_weight': 10, 'i_frequency': 20, 'network_size': 256, 'num_attractors': 4, 'num_cues': 10}"}),
#                               num_attractors, num_cues)
# sweep recalls in [0.1-1] with step 0.1
recall_plot_list = []
ps_frequencies_plot_list = []
spikes_cues_plot_list = []
spikes_recall_plot_list = []
energy_cues_plot_list = []
energy_recalls_plot_list = []
accuracies_plot_list = []
lower_lim = np.arange(0., 1.0, 0.1)
upper_lim = np.arange(0.1, 1.1, 0.1)
for r1, r2 in zip(lower_lim, upper_lim):
    queried_param = df.query(f'accuracy == 1 & recall > {r1} & recall <= {r2}').iloc[0]
    recall_plot_list.append(queried_param['recall'])
    ps_frequencies, spikes_cues, spikes_recall, energy_cues, energy_recalls, returned_params = run_rcn_and_compute_frequency(
        queried_param, num_attractors, num_cues)
    ps_frequencies_plot_list.append(mean(ps_frequencies.values()))
    spikes_cues_plot_list.append(mean(spikes_cues.values()))
    spikes_recall_plot_list.append(mean(spikes_recall.values()))
    energy_cues_plot_list.append(energy_cues)
    energy_recalls_plot_list.append(energy_recalls)
    accuracies_plot_list.append(returned_params)

fig, axes = plt.subplots(3, figsize=(10, 15))
axes[0].scatter(recall_plot_list, ps_frequencies_plot_list)
for i, txt in enumerate(ps_frequencies_plot_list):
    axes[0].annotate(f'{txt:.2f}', (recall_plot_list[i], ps_frequencies_plot_list[i]),
                     xytext=(recall_plot_list[i], ps_frequencies_plot_list[i] + 0.15))
axes[0].set_ylabel('Frequency (Hz)')
axes[0].set_xlabel('Recall')
axes[0].set_title('PS frequency')
axes[1].scatter(recall_plot_list, spikes_cues_plot_list, label='Cue')
for i, txt in enumerate(spikes_cues_plot_list):
    axes[1].annotate(f'{txt:.2f}', (recall_plot_list[i], spikes_cues_plot_list[i]),
                     xytext=(recall_plot_list[i], spikes_cues_plot_list[i] + 100))
axes[1].scatter(recall_plot_list, spikes_recall_plot_list, label='Recall')
for i, txt in enumerate(spikes_recall_plot_list):
    axes[1].annotate(f'{txt:.2f}', (recall_plot_list[i], spikes_recall_plot_list[i]),
                     xytext=(recall_plot_list[i], spikes_recall_plot_list[i] + 100))
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_xlabel('Recall')
axes[1].set_title('Spikes frequency')
axes[1].legend()
axes[2].scatter(recall_plot_list, energy_cues_plot_list, label='Cue')
for i, txt in enumerate(energy_cues_plot_list):
    axes[2].annotate(f'{txt:.2e}', (recall_plot_list[i], energy_cues_plot_list[i]),
                     xytext=(recall_plot_list[i], energy_cues_plot_list[i] + 0.000000001))
axes[2].scatter(recall_plot_list, energy_recalls_plot_list, label='Recall')
for i, txt in enumerate(energy_recalls_plot_list):
    axes[2].annotate(f'{txt:.2e}', (recall_plot_list[i], energy_recalls_plot_list[i]),
                     xytext=(recall_plot_list[i], energy_recalls_plot_list[i] + 0.000000001))
energy_total = [x + y for x, y in zip(energy_cues_plot_list, energy_recalls_plot_list)]
axes[2].scatter(recall_plot_list, energy_total, label='Total')
for i, txt in enumerate(energy_total):
    axes[2].annotate(f'{txt:.2e}', (recall_plot_list[i], energy_total[i]),
                     xytext=(recall_plot_list[i], energy_total[i] + 0.000000001))
axes[2].set_ylabel('Energy (J)')
axes[2].set_xlabel('Recall')
axes[2].set_title('Energy consumption')
axes[2].legend()
fig.tight_layout()
fig.show()
