# -*- coding: utf-8 -*-
"""
@author: t.f.tiotto@rug.nl
@university: University of Groningen
@group: Bernoulli Institute
"""

import atexit
import copy
import glob
import os
import random
import shutil
import signal
from functools import partial

import pathos
import numpy as np
import matplotlib.pyplot as plt
from dill import PicklingWarning, UnpicklingWarning
from natsort import natsorted
from numpy import VisibleDeprecationWarning
import warnings
import pandas as pd

from brian2 import prefs, ms, Hz, mV, second
from tqdm import tqdm

from helper_functions.recurrent_competitive_network import RecurrentCompetitiveNet
from helper_functions.population_spikes import count_ps
from helper_functions.other import *
from plotting_functions.plot_thresholds import *
from plotting_functions.x_u_spks_from_basin import plot_x_u_spks_from_basin

warnings.simplefilter("ignore", PicklingWarning)
warnings.simplefilter("ignore", UnpicklingWarning)
warnings.simplefilter("ignore", VisibleDeprecationWarning)


def cleanup(exit_code=None, frame=None):
    try:
        print("Cleaning up leftover files...")
        # delete tmp folder
        shutil.rmtree(tmp_folder)
    except FileNotFoundError:
        pass


atexit.register(cleanup)
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

tmp_folder = f'tmp_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
os.makedirs(tmp_folder)


def run_sim(params, plot=True, show_plot=False):
    pid = os.getpid()
    print(f'RUNNING worker {pid} with params: {params}')

    ba = params[0]
    i_e_w = params[1]
    i_freq = params[2]
    cue_percentage = params[3]
    cue_time = params[4]
    num_attractors = params[5]
    num_cues = params[6]

    rcn = RecurrentCompetitiveNet(
        plasticity_rule='LR4',
        parameter_set='2.2')

    plastic_syn = False
    plastic_ux = True
    rcn.E_E_syn_matrix_snapshot = False
    rcn.w_e_i = 3 * mV  # for param. 2.1: 5*mV
    rcn.w_max = 10 * mV  # for param. 2.1: 10*mV
    rcn.spont_rate = ba * Hz

    rcn.w_e_i = 3 * mV  # 3 mV default
    rcn.w_i_e = i_e_w * mV  # 1 mV default

    # -- intrinsic plasticity setup (Vth_e_decr for naive / tau_Vth_e for calcium-based)

    rcn.tau_Vth_e = 0.1 * second  # 0.1 s default
    # rcn.Vth_e_init = -51.6 * mV           # -52 mV default
    rcn.k = 4  # 2 default

    rcn.net_init()

    # -- synaptic augmentation setup
    rcn.U = 0.2  # 0.2 default

    rcn.set_stimulus_i(
        stimulus='flat_to_I',
        frequency=i_freq * Hz)  # default: 15 Hz

    # --folder for simulation results
    rcn.net_sim_data_path = make_timestamped_folder(os.path.join(tmp_folder, f'{pid}'))

    # output .txt with simulation summary.
    _f = os.path.join(rcn.net_sim_data_path, 'simulation_summary.txt')

    attractors_list = []
    for i in range(num_attractors):
        if i * (64 + 16) + rcn.stim_size_e > len(rcn.E):
            print(
                f'{num_attractors} attractors of size {rcn.stim_size_e} cannot fit into a network of {len(rcn.E)} neurons.  Instantiating {i} attractors instead.')
            num_attractors = i
            break
        stim_id = rcn.set_active_E_ids(stimulus='flat_to_E_fixed_size', offset=i * (64 + 16))
        rcn.set_potentiated_synapses(stim_id)
        attractors_list.append([f'A{i}', stim_id])

    rcn.set_E_E_plastic(plastic=plastic_syn)
    rcn.set_E_E_ux_vars_plastic(plastic=plastic_ux)

    # generate shuffled attractors
    attractors_cueing_order = [copy.deepcopy(random.choice(attractors_list))]
    for i in range(num_cues - 1):
        atr_sel = copy.deepcopy(random.choice(attractors_list))
        while attractors_cueing_order[i][0] == atr_sel[0]:
            atr_sel = copy.deepcopy(random.choice(attractors_list))
        attractors_cueing_order.append(atr_sel)

    # generate random reactivation times in the range [0, 1]
    for a in attractors_cueing_order:
        a.append(random.uniform(0.5, 5))

    # TODo compute estimated sim time based on cue and activity lengths
    for a in tqdm(attractors_cueing_order, disable=not plot, total=len(attractors_cueing_order)):
        gs = (cue_percentage, a[1], (rcn.net.t / second, rcn.net.t / second + cue_time))
        act_ids = rcn.generic_stimulus(
            frequency=rcn.stim_freq_e,
            stim_perc=gs[0],
            subset=gs[1])
        rcn.run_net(duration=cue_time)
        rcn.generic_stimulus_off(act_ids)
        # # wait for 2 seconds before cueing next attractor
        rcn.run_net(duration=a[2])
        a.append(gs)

    # -- calculate score
    atr_ps_counts = count_ps(rcn=rcn, attractor_cueing_order=attractors_cueing_order)

    trig, spont, score = compute_ps_score(atr_ps_counts)

    if plot:
        title_addition = f'BA {ba} Hz, GS {cue_percentage} %, I-to-E {i_e_w} mV, I input {i_freq} Hz'
        filename_addition = f'_BA_{ba}_GS_{cue_percentage}_W_{i_e_w}_Hz_{i_freq}'

        plot_x_u_spks_from_basin(
            path=rcn.net_sim_data_path,
            filename='x_u_spks_from_basin' + filename_addition,
            title_addition=title_addition,
            rcn=rcn,
            attractor_cues=attractors_cueing_order,
            pss_categorised=atr_ps_counts,
            num_neurons=len(rcn.E),
            show=show_plot)

        # plot_thresholds(
        #     path=rcn.net_sim_data_path,
        #     file_name='thresholds' + filename_addition,
        #     rcn=rcn,
        #     attractors=attractors_list,
        #     show=True)

    print(f'FINISHED worker {pid} triggered: {trig}, spontaneous: {spont}, score: {score}')
    print(atr_ps_counts)

    # TODO do we like this way of scoring?
    return params, score


num_par = 2
cv = 1
default_params = [15, 10, 20, 100, 3, 10]
param_grid = {
    'ba': [default_params[0]],
    'i_e_w': [default_params[1]],
    'i_freq': [default_params[2]],
    'cue_percentage': [default_params[3]],
    'a2_cue_time': np.linspace(0.1, 1, num=num_par),
    'num_attractors': [default_params[4]],
    'num_cues': [default_params[5]],
}
sweeped_params = [(k, i) for i, (k, v) in enumerate(param_grid.items()) if len(v) > 1]

print(param_grid)


def product_dict_to_list(**kwargs):
    from itertools import product

    vals = kwargs.values()

    out_list = []
    for instance in product(*vals):
        out_list.append(list(instance))

    return out_list


# ------ TODO debug
default_params.insert(4, 0.1)
default_params[5] = 3
default_params[6] = 2
p, s = run_sim(default_params, plot=True, show_plot=True)
0 / 0
# ----------------

param_grid_pool = product_dict_to_list(**param_grid) * cv
num_proc = pathos.multiprocessing.cpu_count()
estimate_search_time(run_sim, param_grid_pool, cv)

with pathos.multiprocessing.ProcessPool(num_proc) as p:
    results = p.map(run_sim, param_grid_pool)

if len(sweeped_params) > 1:
    res_unsweeped_removed = []
    for r in results:
        res_tmp = []
        for p in sweeped_params:
            res_tmp.append(r[0][p[1]])
        res_unsweeped_removed.append((res_tmp, r[1]))
else:
    res_unsweeped_removed = results

df_results = pd.DataFrame([i for i in res_unsweeped_removed], columns=['params', 'score'])

df_results_print = df_results.copy()
df_results_print['params'] = df_results['params'].astype(str).apply(lambda x: ''.join(x))
df_results_print = df_results_print.groupby('params').mean().reset_index()
df_results_print.sort_values(by='params', ascending=True, inplace=True)
df_results_print.set_index('params', inplace=True)
df_results_print.index = natsorted(df_results_print.index)
df_results_print.to_csv(f'{tmp_folder}/results.csv')

print(df_results.sort_values(by='score', ascending=False))
# TODO ordering by parameter is still off
fig, ax = plt.subplots(figsize=(12, 10))
df_results.plot(kind='bar', ax=ax)
ax.set_ylabel('Score')
ax.set_xlabel('Parameters')
ax.set_title('Score for different parameters')
fig.savefig(f'{tmp_folder}/score.png')
fig.show()

best_params = df_results.loc[df_results.score.idxmax(), 'params']
print(f'Best parameters: {best_params}')
run_sim(best_params, plot=True, show_plot=True)

while True:
    save = input('Save results? (y/n)')
    if save == 'y':
        save_folder = f'SAVED_({datetime.now().strftime("%Y-%m-%d_%H-%M-%S")})'
        os.makedirs(save_folder)
        os.rename(f'{tmp_folder}/score.png', f'{save_folder}/score.png')
        os.rename(f'{tmp_folder}/results.csv', f'{save_folder}/results.csv')

        save2 = input('Saved overall results.  Also save individual plots?: (y/n)')
        if save2 == 'y':
            for f in glob.glob(f'{tmp_folder}/**/*.png', recursive=True):
                pid = os.path.normpath(f).split(os.path.sep)[1]
                os.makedirs(f'{save_folder}/{pid}', exist_ok=True)
                os.rename(f, f'{save_folder}/{pid}/{os.path.split(f)[1]}')

        cleanup()
        exit()
    elif save == 'n':
        cleanup()
        exit()
    else:
        print("Please enter 'y' or 'n'")
