import atexit
import os
import shutil
import signal
import pathos
import numpy as np
import matplotlib.pyplot as plt
from dill import PicklingWarning, UnpicklingWarning
from natsort import natsorted
from numpy import VisibleDeprecationWarning
import warnings
import pandas as pd

from brian2 import prefs, ms, Hz, mV, second
from helper_functions.recurrent_competitive_network import RecurrentCompetitiveNet
from helper_functions.population_spikes import count_ps
from helper_functions.other import *

warnings.simplefilter("ignore", PicklingWarning)
warnings.simplefilter("ignore", UnpicklingWarning)
warnings.simplefilter("ignore", VisibleDeprecationWarning)

timestamp_folder = make_timestamped_folder('../../../results/RCN_controlled_PS_grid_search_4/')


def cleanup(exit_code=None, frame=None):
    print("Cleaning up leftover files...")
    # delete tmp folder
    shutil.rmtree(timestamp_folder)


atexit.register(cleanup)
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def run_sim(params):
    pid = os.getpid()
    print(f'RUNNING worker {pid} with params: {params}')

    ba = params[0]
    i_e_w = params[1]
    i_freq = params[2]
    cue_percentage = params[3]
    cue_time = params[4]
    num_attractors = params[5]

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
    rcn.net_sim_data_path = make_timestamped_folder(os.path.join(timestamp_folder, f'{pid}'))

    # output .txt with simulation summary.
    _f = os.path.join(rcn.net_sim_data_path, 'simulation_summary.txt')

    attractors_list = []
    for i in range(num_attractors):
        if i * 80 + rcn.stim_size_e > len(rcn.E):
            print(
                f'{num_attractors} attractors of size {rcn.stim_size_e} cannot fit into a network of {len(rcn.E)} neurons.  Instantiating {i} attractors instead.')
            num_attractors = i
            break
        stim_id = rcn.set_active_E_ids(stimulus='flat_to_E_fixed_size', offset=i * 80)
        rcn.set_potentiated_synapses(stim_id)
        attractors_list.append([f'A{i}', stim_id])

    rcn.set_E_E_plastic(plastic=plastic_syn)
    rcn.set_E_E_ux_vars_plastic(plastic=plastic_ux)

    for i, a in enumerate(attractors_list):
        gs = (cue_percentage, a[1], ((2 + cue_time) * i, (2 + cue_time) * i + cue_time))
        act_ids = rcn.generic_stimulus(
            frequency=rcn.stim_freq_e,
            stim_perc=gs[0],
            subset=gs[1])
        rcn.run_net(duration=gs[2][1] - gs[2][0])
        rcn.generic_stimulus_off(act_ids)
        # # wait for 2 seconds before cueing second attractor
        rcn.run_net(duration=2)
        a.append(gs[2])

    attractors_t_windows = []
    for i, a in enumerate(attractors_list):
        try:
            attractors_t_windows.append((attractors_list[i][2][1], attractors_list[i + 1][2][0], attractors_list[i][2]))
        except IndexError:
            attractors_t_windows.append((attractors_list[i][2][1], rcn.net.t / second, attractors_list[i][2]))

    # -- calculate score
    atr_ps_counts = count_ps(
        rcn=rcn,
        attractors=attractors_list,
        time_window=attractors_t_windows,
        spk_sync_thr=0.75)

    # -- count reactivations
    trig = 0
    spont = 0
    for k, v in atr_ps_counts.items():
        for k, v in v.items():
            if k == 'triggered':
                trig += v
            if k == 'spontaneous':
                spont += v
    try:
        score = trig / (trig + spont)
    except ZeroDivisionError:
        score = 0

    print(f'FINISHED worker {pid} triggered: {trig}, spontaneous: {spont}, score: {score}')

    # TODO do we like this way of scoring?
    return params, score


num_par = 10
cv = 20
default_params = [15, 10, 20, 100, 2]
param_grid = {
    'ba': [default_params[0]],
    'i_e_w': [default_params[1]],
    'i_freq': [default_params[2]],
    'cue_percentage': [default_params[3]],
    'a2_cue_time': np.linspace(0.1, 1, num=num_par),
    'num_attractors': [default_params[4]]
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


param_grid_pool = product_dict_to_list(**param_grid) * cv
num_proc = pathos.multiprocessing.cpu_count()
estimate_search_time(run_sim, param_grid_pool, cv)

with pathos.multiprocessing.ProcessPool(num_proc) as p:
    results = p.map(run_sim, param_grid_pool)

res_unsweeped_removed = []
for r in results:
    res_tmp = []
    for p in sweeped_params:
        res_tmp.append(r[0][p[1]])
    res_unsweeped_removed.append((res_tmp, r[1]))

df_results = pd.DataFrame([i for i in res_unsweeped_removed], columns=['params', 'score'])
df_results['params'] = df_results['params'].astype(str).apply(lambda x: ''.join(x))
df_results = df_results.groupby('params').mean().reset_index()
df_results.sort_values(by='params', ascending=True, inplace=True)
df_results.set_index('params', inplace=True)
df_results.index = natsorted(df_results.index)

print(df_results.sort_values(by='score', ascending=False))
# TODO ordering by parameter is still off
fig, ax = plt.subplots(figsize=(12, 10))
df_results.plot(kind='bar', ax=ax)
ax.set_ylabel('Score')
ax.set_xlabel('Parameters')
ax.set_title('Score for different parameters')
plt.show()

cleanup()
