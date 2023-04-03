# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System / Artificial Intelligence
"""

# python3 RCN_conditional_activation.py --attractors 3

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

print_report = True

if sys.platform == 'linux':

    root = os.path.dirname(os.path.abspath(os.path.join(__file__ , '../../..')))

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

prefs.codegen.target = 'numpy'

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
warnings.filterwarnings('ignore', category=TqdmWarning)


# === parsing arguments =================================================================

parser = argparse.ArgumentParser(description='RCN_item_reactivation_gs_attractors')

parser.add_argument('--ba_rate', type=float, default=15, help='Background mean rate (Hz).')

parser.add_argument('--gs_percent', type=float, default=100, help='Percentage of stimulated neurons in attractor.')

parser.add_argument('--W_ie', type=float, default=10, help='I-to-E inhibition weight.')

parser.add_argument('--inh_rate', type=float, default=20, help='Inhibition mean rate (Hz).')

parser.add_argument('--attractors', type=int, default=2, choices=[1, 2, 3], help='Number of attractors')

parser.add_argument('--A1_setting', type=float, default=(0.0, 2.0, 0.1), nargs='+', help='Attractor sim. setting (A, B, C), with A = start, B = end, C = cue time.')

parser.add_argument('--A2_setting', type=float, default=(2.0, 4.0, 0.7), nargs='+', help='Attractor sim. setting (A, B, C), with A = start, B = end, C = cue time.')

parser.add_argument('--cue_A1', type=int, default=1, help='')
parser.add_argument('--cue_A2', type=int, default=1, help='')

args = parser.parse_args()


# === simulation parameters =============================================================

if print_report:

    print('''\n
    In this experiment the conditional activation of two attractors is simulated.
    If attractor A2 is cued after attractor A1 has been active, both A1 and A2 become concurrently active.
    On the other hand, if A1 is cued after A2 has been active, then only A1 stays active.''')

timestamp_folder = make_timestamped_folder('../../../results/RCN_conditional_activation/')

plasticity_rule = 'LR4'
parameter_set = '2.2'


# === initializing/running network ======================================================
    
# --initialise time to predict overall experiment time

rcn = RecurrentCompetitiveNet(
    plasticity_rule = plasticity_rule,
    parameter_set = parameter_set)

plastic_syn = False
plastic_ux = True
rcn.E_E_syn_matrix_snapshot = False
rcn.w_e_i = 3 * mV                      # for param. 2.1: 5*mV
rcn.w_max = 10 * mV                     # for param. 2.1: 10*mV
rcn.spont_rate = args.ba_rate * Hz

rcn.w_e_i = 3 * mV                      # 3 mV default
rcn.w_i_e = args.W_ie * mV                  # 1 mV default

# -- intrinsic plasticity setup (Vth_e_decr for naive / tau_Vth_e for calcium-based)

rcn.tau_Vth_e = 0.1 * second            # 0.1 s default
rcn.k = 4                               # 2 default

rcn.net_init()

# -- synaptic augmentation setup
rcn.U = 0.2                             # 0.2 default

rcn.set_stimulus_i(
    stimulus = 'flat_to_I', 
    frequency = args.inh_rate * Hz)            # default: 15 Hz

# --folder for simulation results
save_dir = os.path.join(os.path.join(timestamp_folder, f'BA_{args.ba_rate}_GS_{args.gs_percent}_W_{args.W_ie}_I_{args.inh_rate}'))
os.mkdir(save_dir)
rcn.net_sim_data_path = save_dir

# output .txt with simulation summary.
_f = os.path.join(rcn.net_sim_data_path, 'simulation_summary.txt')

attractors = []

if args.attractors >= 1:
    stim1_ids = rcn.set_active_E_ids(stimulus = 'flat_to_E_fixed_size', offset = 0)
    rcn.set_potentiated_synapses(stim1_ids)
    A1 = list(range(0, 64))
    attractors.append(('A1', A1))
if args.attractors >= 2:
    stim2_ids = rcn.set_active_E_ids(stimulus = 'flat_to_E_fixed_size', offset = 100)
    rcn.set_potentiated_synapses(stim2_ids)
    A2 = list(range(100, 164))
    attractors.append(('A2', A2))
if args.attractors >= 3:
    stim3_ids = rcn.set_active_E_ids(stimulus = 'flat_to_E_fixed_size', offset = 180)
    rcn.set_potentiated_synapses(stim3_ids)
    A3 = list(range(180, 244))
    attractors.append(('A3', A3))

rcn.set_synapses_A_2_B(A_ids = stim2_ids, B_ids = stim1_ids, weight = 1.5*mV) # connects A2 to A1.

# rcn.set_synapses_A_2_GO(A_ids = stim1_ids, GO_ids = stim3_ids, weight = 1*mV)
# rcn.set_synapses_A_2_GO(A_ids = stim2_ids, GO_ids = stim3_ids, weight = 1*mV)

rcn.set_E_E_plastic(plastic = plastic_syn)
rcn.set_E_E_ux_vars_plastic(plastic = plastic_ux)

# -- run network --

def run_sim(A1_setting, A2_setting):

    if A1_setting[0] < A2_setting[0]:

        # --- cue attractor A1 ---
        act_ids = rcn.generic_stimulus(
            frequency = rcn.stim_freq_e, 
            stim_perc = args.gs_percent, 
            subset = stim1_ids)
        
        if args.cue_A1:

            rcn.run_net(duration = A1_setting[2])
        
        rcn.generic_stimulus_off(act_ids)

        # --- attractor A1 free activity ---
        rcn.run_net(duration = A1_setting[1] - A1_setting[0])

        # --- cue attractor A2 ---
        act_ids = rcn.generic_stimulus(
            frequency = rcn.stim_freq_e, 
            stim_perc = args.gs_percent, 
            subset = stim2_ids)

        if args.cue_A2:

            rcn.run_net(duration = A2_setting[2])
        
        rcn.generic_stimulus_off(act_ids)

        # --- attractor A2 free activity ---
        rcn.run_net(duration = A2_setting[1] - A2_setting[0])

    else:

        # --- cue attractor A2 ---
        act_ids = rcn.generic_stimulus(
            frequency = rcn.stim_freq_e, 
            stim_perc = args.gs_percent, 
            subset = stim2_ids)

        if args.cue_A2:

            rcn.run_net(duration = A2_setting[2])
        
        rcn.generic_stimulus_off(act_ids)

        # --- attractor A2 free activity ---
        rcn.run_net(duration = A2_setting[1] - A2_setting[0])

        # --- cue attractor A1 ---
        act_ids = rcn.generic_stimulus(
            frequency = rcn.stim_freq_e, 
            stim_perc = args.gs_percent, 
            subset = stim1_ids)
        
        if args.cue_A1:

            rcn.run_net(duration = A1_setting[2])
        
        rcn.generic_stimulus_off(act_ids)

        # --- attractor A1 free activity ---
        rcn.run_net(duration = A1_setting[1] - A1_setting[0])

run_sim(args.A1_setting, args.A2_setting)

# 2 ------ plotting simulation data ------

title_addition = f'BA {args.ba_rate} Hz, GS {args.gs_percent} %, I-to-E {args.W_ie} mV, I input {args.inh_rate} Hz'
filename_addition = f'_BA_{args.ba_rate}_GS_{args.gs_percent}_W_{args.W_ie}_Hz_{args.inh_rate}'

fig1 = plot_x_u_spks_from_basin(
    path = save_dir,
    filename = 'x_u_spks_from_basin' + filename_addition,
    title_addition = title_addition,
    rcn = rcn,
    attractors = attractors,
    num_neurons = len(rcn.E),
    show = False)

rcn.export_attractors_data(attractor_A = A1, attractor_B = A2)