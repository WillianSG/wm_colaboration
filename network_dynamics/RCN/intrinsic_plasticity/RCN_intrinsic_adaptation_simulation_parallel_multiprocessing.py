import atexit
import os
import shutil
import signal
import pathos
import numpy as np
import matplotlib.pyplot as plt
from dill import PicklingWarning, UnpicklingWarning
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
    a2_cue_time = params[4]
    attractors = params[5]

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
    if attractors >= 1:
        stim1_ids = rcn.set_active_E_ids(stimulus='flat_to_E_fixed_size', offset=0)
        rcn.set_potentiated_synapses(stim1_ids)
        A1 = list(range(0, 64))
        attractors_list.append(('A1', A1))
    if attractors >= 2:
        stim2_ids = rcn.set_active_E_ids(stimulus='flat_to_E_fixed_size', offset=100)
        rcn.set_potentiated_synapses(stim2_ids)
        A2 = list(range(100, 164))
        attractors_list.append(('A2', A2))
    if attractors >= 3:
        stim3_ids = rcn.set_active_E_ids(stimulus='flat_to_E_fixed_size', offset=180)
        rcn.set_potentiated_synapses(stim3_ids)
        A3 = list(range(180, 244))
        attractors_list.append(('A3', A3))

    rcn.set_E_E_plastic(plastic=plastic_syn)
    rcn.set_E_E_ux_vars_plastic(plastic=plastic_ux)

    stimulation_amount = []

    # cue attractor 1
    gs_A1 = (cue_percentage, stim1_ids, (0, 0.1))
    act_ids = rcn.generic_stimulus(
        frequency=rcn.stim_freq_e,
        stim_perc=gs_A1[0],
        subset=gs_A1[1])
    stimulation_amount.append(
        (100 * np.intersect1d(act_ids, stim1_ids).size / len(stim1_ids),
         100 * np.intersect1d(act_ids, stim2_ids).size / len(stim2_ids))
    )
    rcn.run_net(duration=gs_A1[2][1] - gs_A1[2][0])
    rcn.generic_stimulus_off(act_ids)

    # wait for 2 seconds before cueing second attractor
    rcn.run_net(duration=2)

    # cue attractor 2
    gs_A2 = (cue_percentage, stim2_ids, (2.1, 2.1 + a2_cue_time))
    act_ids = rcn.generic_stimulus(
        frequency=rcn.stim_freq_e,
        stim_perc=gs_A2[0],
        subset=gs_A2[1])
    stimulation_amount.append(
        (100 * np.intersect1d(act_ids, stim1_ids).size / len(stim1_ids),
         100 * np.intersect1d(act_ids, stim2_ids).size / len(stim2_ids))
    )
    rcn.run_net(duration=gs_A2[2][1] - gs_A2[2][0])
    rcn.generic_stimulus_off(act_ids)

    # wait for another 2 seconds before ending
    rcn.run_net(duration=2 + a2_cue_time)

    attractors_t_windows = []
    # log period of independent activity for attractor 1.
    attractors_t_windows.append((gs_A1[2][1], gs_A2[2][0], gs_A1[2]))
    # log period of independent activity for attractor 2.
    attractors_t_windows.append((gs_A2[2][1], rcn.E_E_rec.t[-1] / second, gs_A2[2]))

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


num_par = 20
cv = 10
default_params = [15, 10, 20, 100, 2]
param_grid = {
    'ba': [default_params[0]],
    'i_e_w': [default_params[1]],
    'i_freq': [default_params[2]],
    'cue_percentage': [default_params[3]],
    'a2_cue_time': np.linspace(0.1, 1, num=num_par),
    'attractors': [default_params[4]]
}
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

df_results = pd.DataFrame(results, columns=['params', 'score'])
df_results['params'] = df_results['params'].astype(str).apply(lambda x: ''.join(x))
df_results = df_results.groupby('params').mean().reset_index()
df_results.set_index('params', inplace=True)

print(df_results.sort_values(by='score', ascending=False))

fig, ax = plt.subplots(figsize=(12, 10))
df_results.plot(kind='bar', ax=ax)
ax.set_ylabel('Score')
ax.set_xlabel('Parameters')
ax.set_title('Score for different parameters')
plt.show()

cleanup()
