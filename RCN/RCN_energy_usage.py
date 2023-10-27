import ast
import os
from collections import defaultdict
from datetime import datetime
from statistics import mean

import pandas as pd
import numpy as np
from random import randrange

from matplotlib import pyplot as plt

from helper_functions.recurrent_competitive_network import run_rcn


def run_rcn_and_compute_frequency(params, num_attractors=2, num_cues=2):
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

    rcn, returned_params = run_rcn(params, show_plot=False,
                                   progressbar=True, low_memory=False,
                                   attractor_conflict_resolution="3", return_complete_stats=True,
                                   seed_init=seed)

    print('\tPS frequency')
    ps_frequencies = {}
    for k, v in returned_params['ps_counts'].items():
        ps_freq = len(v['triggered']) / v['cued_time']
        ps_frequencies[k] = ps_freq
        print(f"\t\t{k}: {ps_freq} Hz")

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
        print('\t\tCued frequency')
        for k, v in spikes_cues.items():
            spikes_cues[k] = np.mean(v)
            print(f"\t\t\t{k}: {spikes_cues[k]} Hz")
        print('\t\tRecall frequency')
        for k, v in spikes_recall.items():
            spikes_recall[k] = np.mean(v)
            print(f"\t\t\t{k}: {spikes_recall[k]} Hz")

        return spikes_cues, spikes_recall

    print('\tExcitatory spikes frequency')
    spk_mon_ids_e, spk_mon_ts_e = rcn.get_spks_from_pattern_neurons()
    freq_cues_e, freq_recall_e = get_spikes_frequency(spk_mon_ts_e)

    print('\tInhibitory spikes frequency')
    spk_mon_ids_i, spk_mon_ts_i = rcn.get_I_spks()
    freq_cues_i, freq_recall_i = get_spikes_frequency(spk_mon_ts_i)

    del rcn

    return ps_frequencies, freq_cues_e, freq_recall_e, freq_cues_i, freq_recall_i, returned_params


save_folder = f'RESULTS/EFFICIENCY_SAVED_({datetime.now().strftime("%Y-%m-%d_%H-%M-%S")})'
os.makedirs(save_folder)
print("TMP:", save_folder)

num_attractors = 2
num_cues = 100
seed = randrange(1000000)
# load results from RCN Bayesian sweep
df = pd.read_csv("RESULTS/BAYESIAN_OPTIMISATION/4ATR_SWEEP_(2023-08-31_00-39-08)/results_min_max_scaled.csv")
# original parameters
# run_rcn_and_compute_frequency(pd.Series({'score': -1, 'accuracy': -1, 'recall': -1,
#                                          'params': "{'attractor_size': 64, 'background_activity': 15, 'cue_length': 1, 'cue_percentage': 100, 'e_e_max_weight': 10, 'e_i_weight': 3, 'i_e_weight': 10, 'i_frequency': 20, 'network_size': 256, 'num_attractors': 4, 'num_cues': 10}"}),
#                               num_attractors, num_cues)
# sweep recalls in [0.1-1] with step 0.1
recall_plot_list = []
ps_frequencies_plot_list = []
spikes_cues_plot_list_e = []
spikes_recall_plot_list_e = []
spikes_cues_plot_list_i = []
spikes_recall_plot_list_i = []
energy_cues_plot_list = []
energy_recalls_plot_list = []
accuracies_plot_list = []
lower_lim = np.arange(0., 1.0, 0.1)
upper_lim = np.arange(0.1, 1.1, 0.1)
# lower_lim = [0.8, 0.9]
# upper_lim = [0.9, 1.0]
for r1, r2 in zip(lower_lim, upper_lim):
    queried_param = df.query(f'accuracy == 1 & recall > {r1} & recall <= {r2}').iloc[0]
    recall_plot_list.append(queried_param['recall'])
    ps_frequencies, freq_cues_e, freq_recall_e, freq_cues_i, freq_recall_i, returned_params = run_rcn_and_compute_frequency(
        queried_param, num_attractors, num_cues)
    ps_frequencies_plot_list.append(mean(ps_frequencies.values()))
    spikes_cues_plot_list_e.append(mean(freq_cues_e.values()))
    spikes_recall_plot_list_e.append(mean(freq_recall_e.values()))
    spikes_cues_plot_list_i.append(mean(freq_cues_i.values()))
    spikes_recall_plot_list_i.append(mean(freq_recall_i.values()))
    accuracies_plot_list.append(returned_params)

results = pd.DataFrame({'Recall': recall_plot_list,
                        'Frequency Cue E': spikes_cues_plot_list_e,
                        'Frequency Recall E': spikes_recall_plot_list_e,
                        'Frequency Cue I': spikes_cues_plot_list_i,
                        'Frequency Recall I': spikes_recall_plot_list_i,
                        'PS Frequency': ps_frequencies_plot_list},
                       index=recall_plot_list)
results.to_csv(f'{save_folder}/results.csv')


def annotate_plot(ax, spikes_plot_list):
    shift = np.mean(spikes_plot_list) / 1000

    for i, txt in enumerate(spikes_plot_list):
        ax.annotate(f'{txt:.2f}', (recall_plot_list[i], spikes_plot_list[i]),
                    xytext=(recall_plot_list[i], spikes_plot_list[i] + shift))


fig, axes = plt.subplots(2, figsize=(10, 15))
axes[0].scatter(recall_plot_list, ps_frequencies_plot_list)
annotate_plot(axes[0], ps_frequencies_plot_list)
axes[0].set_ylabel('Frequency (Hz)')
axes[0].set_xlabel('Recall')
axes[0].set_title('PS frequency')
axes[1].scatter(recall_plot_list, spikes_cues_plot_list_e, color='b', marker=',', label='Cue (E)')
axes[1].scatter(recall_plot_list, spikes_cues_plot_list_i, color='r', marker=',', label='Cue (I)')
annotate_plot(axes[1], spikes_cues_plot_list_e)
annotate_plot(axes[1], spikes_cues_plot_list_i)
axes[1].scatter(recall_plot_list, spikes_recall_plot_list_e, color='b', label='Recall (E)')
axes[1].scatter(recall_plot_list, spikes_recall_plot_list_i, color='r', label='Recall (I)')
annotate_plot(axes[1], spikes_recall_plot_list_e)
annotate_plot(axes[1], spikes_recall_plot_list_i)
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_xlabel('Recall')
axes[1].set_title('Spikes frequency')
axes[1].legend()
fig.tight_layout()
fig.show()
fig.savefig(f'{save_folder}/recall_vs_efficiency.png')
