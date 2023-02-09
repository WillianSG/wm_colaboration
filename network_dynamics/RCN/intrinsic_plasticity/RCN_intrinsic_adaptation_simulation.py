# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl / t.f.tiotto@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System / Artificial Intelligence
"""

import itertools
import os, sys, pickle, shutil
import matplotlib.pyplot as plt
import numpy as np
import argparse
import multiprocessing as mp
import warnings
from tqdm import TqdmWarning
import timeit
from itertools import product
from brian2 import prefs, ms, Hz, mV, second

print_report = False

if sys.platform == 'linux':

    root = os.path.dirname(os.path.abspath(os.path.join(__file__, '../../..')))

    sys.path.append(os.path.join(root, 'helper_functions'))
    sys.path.append(os.path.join(root, 'plotting_functions'))

    from recurrent_competitive_network import RecurrentCompetitiveNet
    from other import *
    from x_u_spks_from_basin import plot_x_u_spks_from_basin
    from rcn_spiketrains_histograms import plot_rcn_spiketrains_histograms
    from spike_synchronisation import *
    from plot_thresholds import *
    from population_spikes import count_ps

else:
    from brian2 import prefs, ms, Hz, mV, second
    from helper_functions.recurrent_competitive_network import RecurrentCompetitiveNet
    from helper_functions.other import *
    from plotting_functions.x_u_spks_from_basin import plot_x_u_spks_from_basin
    from plotting_functions.rcn_spiketrains_histograms import plot_rcn_spiketrains_histograms
    from plotting_functions.spike_synchronisation import *
    from plotting_functions.plot_thresholds import *
    from helper_functions.population_spikes import count_ps

prefs.codegen.target = 'numpy'

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
warnings.filterwarnings('ignore', category=TqdmWarning)

# === parsing arguments =================================================================

parser = argparse.ArgumentParser(description='RCN_item_reactivation_gs_attractors')
parser.add_argument('--ba_amount', type=float, default=[15, 15], nargs='+',
                    help='Bounds for background activity in Hz')
parser.add_argument('--ba_step', type=float, default=5, help='Step size for background activity')
parser.add_argument('--gs_amount', type=float, default=[15, 15], nargs='+',
                    help='Bounds for generic stimulus in % of stimulated neurons')
parser.add_argument('--gs_step', type=float, default=5, help='Step size for generic stimulus')
parser.add_argument('--i_amount', type=float, default=[10, 10], nargs='+',
                    help='Bounds for I-to-E inhibition weight')
parser.add_argument('--i_step', type=float, default=2, help='Step size for I-to-E inhibition weight')
parser.add_argument('--i_stim_amount', type=float, default=[20, 20], nargs='+',
                    help='Bounds for I inhibition input in Hz')
parser.add_argument('--i_stim_step', type=int, default=10, help='Step size for I inhibition')
parser.add_argument('--gs_freq', type=float, default=4, help='Frequency of generic stimulus Hz')
parser.add_argument('--gs_length', type=float, default=0.1, help='Length of generic stimulus in seconds')
parser.add_argument('--pre_runtime', type=float, default=0.0, help='Runtime before showing generic stimulus')
parser.add_argument('--gs_runtime', type=float, default=15, help='Runtime for showing generic stimulus')
parser.add_argument('--post_runtime', type=float, default=0.0, help='Runtime after showing generic stimulus')
parser.add_argument('--attractors', type=int, default=2, choices=[1, 2, 3], help='Number of attractors')
parser.add_argument('--show', default=False, action=argparse.BooleanOptionalAction)

parser.add_argument('--A2_cue_time', type=float, default=0.1, help='Duration of cueing')

args = parser.parse_args()

if len(args.ba_amount) == 1:
    args.ba_amount = [args.ba_amount[0], args.ba_amount[0]]
elif len(args.ba_amount) == 2:
    pass
else:
    raise ValueError('Wrong number of background activity amounts')
if len(args.gs_amount) == 1:
    args.gs_amount = [args.gs_amount[0], args.gs_amount[0]]
elif len(args.gs_amount) == 2:
    pass
else:
    raise ValueError('Wrong number of generic stimulus amounts')
if len(args.i_amount) == 1:
    args.i_amount = [args.i_amount[0], args.i_amount[0]]
elif len(args.i_amount) == 2:
    pass
else:
    raise ValueError('Wrong number of I-to-E weight amounts')
if len(args.i_stim_amount) == 1:
    args.i_stim_amount = [args.i_stim_amount[0], args.i_stim_amount[0]]
elif len(args.i_stim_amount) == 2:
    pass
else:
    raise ValueError('Wrong number of I input frequency amounts')

assert args.ba_amount[0] <= args.ba_amount[
    1], 'Lower bound for background activity Hz must be smaller than upper bound'
assert args.gs_amount[0] <= args.gs_amount[1], 'Lower bound for generic stimulus % must be smaller than upper bound'
assert args.gs_freq > 0, 'Generic stimulus frequency must be positive'
assert args.ba_step > 0, 'Step size must be positive'
assert args.gs_step > 0, 'Step size must be positive'
assert args.i_step > 0, 'Step size must be positive'
assert args.i_stim_step > 0, 'Step size must be positive'

# === simulation parameters =============================================================

if print_report:
    print('This experiment will run around 4 seconds of simulation.\n'
          'In the first 2 seconds it will cue attractor 1 and see if it naturally reactivates.\n'
          'In the last 2 seconds it will cue attractor 2 and see if it naturally reactivates.\n'
          'This is to test if STSP and intrinsic plasticity can enable phasic cued recall.')

timestamp_folder = make_timestamped_folder('../../../results/RCN_controlled_PS_grid_search_4/')

plasticity_rule = 'LR4'
parameter_set = '2.2'

# -- sweep over all combinations of parameters
background_activity = np.arange(args.ba_amount[0], args.ba_amount[1] + args.ba_step, args.ba_step)

generic_stimuli = np.arange(args.gs_amount[0], args.gs_amount[1] + args.gs_step, args.gs_step)

I_E_weights = np.arange(args.i_amount[0], args.i_amount[1] + args.i_step, args.i_step)

I_input_freq = np.arange(args.i_stim_amount[0], args.i_stim_amount[1] + args.i_stim_step, args.i_stim_step)

parameter_combinations = itertools.product(background_activity, generic_stimuli, I_E_weights, I_input_freq)

# === initializing/running network ======================================================

i = 0

running_times = []

attractors_t_windows = []

for ba, gs_percentage, i_e_w, i_freq in parameter_combinations:

    # --initialise time to predict overall experiment time

    start_time = timeit.default_timer()

    i += 1

    rcn = RecurrentCompetitiveNet(
        plasticity_rule=plasticity_rule,
        parameter_set=parameter_set)

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
    save_dir = os.path.join(os.path.join(timestamp_folder, f'BA_{ba}_GS_{gs_percentage}_W_{i_e_w}_I_{i_freq}'))
    os.mkdir(save_dir)
    rcn.net_sim_data_path = save_dir

    # output .txt with simulation summary.
    _f = os.path.join(rcn.net_sim_data_path, 'simulation_summary.txt')

    attractors = []

    if args.attractors >= 1:
        stim1_ids = rcn.set_active_E_ids(stimulus='flat_to_E_fixed_size', offset=0)
        rcn.set_potentiated_synapses(stim1_ids)
        A1 = list(range(0, 64))
        attractors.append(('A1', A1))
    if args.attractors >= 2:
        stim2_ids = rcn.set_active_E_ids(stimulus='flat_to_E_fixed_size', offset=100)
        rcn.set_potentiated_synapses(stim2_ids)
        A2 = list(range(100, 164))
        attractors.append(('A2', A2))
    if args.attractors >= 3:
        stim3_ids = rcn.set_active_E_ids(stimulus='flat_to_E_fixed_size', offset=180)
        rcn.set_potentiated_synapses(stim3_ids)
        A3 = list(range(180, 244))
        attractors.append(('A3', A3))

    rcn.set_E_E_plastic(plastic=plastic_syn)
    rcn.set_E_E_ux_vars_plastic(plastic=plastic_ux)

    if print_report:
        print('-------------------------------------------------------')
        print(
            f'Iteration {i} of {len(list(itertools.product(background_activity, generic_stimuli, I_E_weights, I_input_freq)))}',
            end=' : ')
        print('ba = ', ba, 'Hz', end=' , ')
        print('gs = ', gs_percentage, '%', end=' , ')
        print('I input = ', i_freq, 'Hz', 'I-to-E weight', i_e_w, 'mV')

    # -- run network --

    stimulation_amount = []
    cue_percentage = 100

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
    gs_A2 = (cue_percentage, stim2_ids, (2.1, 2.1 + args.A2_cue_time))
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
    rcn.run_net(duration=2 + args.A2_cue_time)

    # 2 ------ plotting simulation data ------
    title_addition = f'BA {ba} Hz, GS {gs_percentage} %, I-to-E {i_e_w} mV, I input {i_freq} Hz'
    filename_addition = f'_BA_{ba}_GS_{gs_percentage}_W_{i_e_w}_Hz_{i_freq}'

    gss = [gs_A1, gs_A2]
    fig1 = plot_x_u_spks_from_basin(
        path=save_dir,
        filename='x_u_spks_from_basin' + filename_addition,
        title_addition=title_addition,
        generic_stimuli=gss,
        rcn=rcn,
        attractors=attractors,
        num_neurons=len(rcn.E),
        show=args.show)

    fig3 = plot_thresholds(
        path=save_dir,
        file_name='thresholds' + filename_addition,
        rcn=rcn,
        attractors=attractors,
        show=args.show)

    # plot u from synapses and neurons

    running_times.append(timeit.default_timer() - start_time)

    if print_report:
        print(
            f'Predicted time remaining in experiment: '
            f'{np.round(np.mean(running_times) * len(list(itertools.product(background_activity, generic_stimuli, I_E_weights, I_input_freq)))) - i} s')

        print('DEBUG\n% neurons stimulated by GS in each attractor\n', stimulation_amount, '\n')

    # log period of independent activity for attractor 1.
    attractors_t_windows.append((gs_A1[2][1], gs_A2[2][0], gs_A1[2]))

    # log period of independent activity for attractor 2.
    attractors_t_windows.append((gs_A2[2][1], rcn.E_E_rec.t[-1] / second, gs_A2[2]))

    with open(_f, 'a') as params_file:

        params_file.write('background activity (Hz): {}\n'.format(ba))
        params_file.write('generic stimuli (%): {}\n'.format(gs_percentage))
        params_file.write('inh. to exc. weight (mV): {}\n'.format(i_e_w))
        params_file.write('inh. firing frequency (Hz): {}\n'.format(i_freq))
        params_file.write('cueing time (s): {}\n'.format(args.A2_cue_time))
        params_file.write('\nPopulation Spikes (count)\n')

atr_ps_counts = count_ps(
    rcn=rcn,
    attractors=attractors,
    time_window=attractors_t_windows,
    spk_sync_thr=0.75)

for key, val in atr_ps_counts.items():
    with open(_f, 'a') as params_file:
        params_file.write('attractor {}: {}\n'.format(key, val))

title_addition = f'BA {ba} Hz, GS {gs_percentage} %, I-to-E {i_e_w} mV, I input {i_freq} Hz'
filename_addition = f'_BA_{ba}_GS_{gs_percentage}_W_{i_e_w}_Hz_{i_freq}'

gss = [gs_A1, gs_A2]
fig1 = plot_x_u_spks_from_basin(
    path=save_dir,
    filename='x_u_spks_from_basin' + filename_addition,
    title_addition=title_addition,
    generic_stimuli=gss,
    rcn=rcn,
    attractors=attractors,
    num_neurons=len(rcn.E),
    show=args.show)
# plot_syn_matrix_heatmap( path_to_data=rcn.E_E_syn_matrix_path )

fig2 = plot_rcn_spiketrains_histograms(
    Einp_spks=rcn.get_Einp_spks()[0],
    Einp_ids=rcn.get_Einp_spks()[1],
    stim_E_size=rcn.stim_size_e,
    E_pop_size=rcn.N_input_e,
    Iinp_spks=rcn.get_Iinp_spks()[0],
    Iinp_ids=rcn.get_Iinp_spks()[1],
    stim_I_size=rcn.stim_size_i,
    I_pop_size=rcn.N_input_i,
    E_spks=rcn.get_E_spks()[0],
    E_ids=rcn.get_E_spks()[1],
    I_spks=rcn.get_I_spks()[0],
    I_ids=rcn.get_I_spks()[1],
    t_run=rcn.net.t,
    path=save_dir,
    filename='rcn_population_spiking' + filename_addition,
    title_addition=title_addition,
    show=args.show)
