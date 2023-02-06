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

    root = os.path.dirname(os.path.abspath(os.path.join(__file__ , '../../..')))

    sys.path.append(os.path.join(root, 'helper_functions'))
    sys.path.append(os.path.join(root, 'plotting_functions'))

    from recurrent_competitive_network import RecurrentCompetitiveNet
    from other import *
    from x_u_spks_from_basin import plot_x_u_spks_from_basin
    from rcn_spiketrains_histograms import plot_rcn_spiketrains_histograms
    from spike_synchronisation import *
    from visualize_attractor import *
    from population_spikes import find_gap_between_PS

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
parser.add_argument('--ba_amount', type=float, default=[10, 10], nargs='+',
                    help='Bounds for background activity in Hz')
parser.add_argument('--ba_step', type=float, default=5, help='Step size for background activity')
parser.add_argument('--gs_amount', type=float, default=[15, 15], nargs='+',
                    help='Bounds for generic stimulus in % of stimulated neurons')
parser.add_argument('--gs_step', type=float, default=5, help='Step size for generic stimulus')
parser.add_argument('--i_amount', type=float, default=[15, 15], nargs='+',
                    help='Bounds for I-to-E inhibition weight')
parser.add_argument('--i_step', type=float, default=2, help='Step size for I-to-E inhibition weight')
parser.add_argument('--i_stim_amount', type=float, default=[15, 15], nargs='+',
                    help='Bounds for I inhibition input in Hz')
parser.add_argument('--i_stim_step', type=int, default=10, help='Step size for I inhibition')
parser.add_argument('--gs_freq', type=float, default=4, help='Frequency of generic stimulus Hz')
parser.add_argument('--gs_length', type=float, default=0.1, help='Length of generic stimulus in seconds')
parser.add_argument('--pre_runtime', type=float, default=0.0, help='Runtime before showing generic stimulus')
parser.add_argument('--gs_runtime', type=float, default=15, help='Runtime for showing generic stimulus')
parser.add_argument('--post_runtime', type=float, default=0.0, help='Runtime after showing generic stimulus')
parser.add_argument('--attractors', type=int, default=2, choices=[1, 2, 3], help='Number of attractors')
parser.add_argument('--show', default=False, action=argparse.BooleanOptionalAction)

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

timestamp_folder = make_timestamped_folder('../../../results/RCN_controlled_PS_Wee/')

plasticity_rule = 'LR4'
parameter_set = '2.2'

# -- sweep over all combinations of parameters
background_activity = np.arange(args.ba_amount[0], args.ba_amount[1] + args.ba_step, args.ba_step)

generic_stimuli = np.arange(args.gs_amount[0], args.gs_amount[1] + args.gs_step, args.gs_step)

I_E_weights = np.arange(args.i_amount[0], args.i_amount[1] + args.i_step, args.i_step)

I_input_freq = np.arange(args.i_stim_amount[0], args.i_stim_amount[1] + args.i_stim_step, args.i_stim_step)

parameter_combinations = itertools.product(background_activity, generic_stimuli, I_E_weights, I_input_freq)


# === initializing/running network ======================================================

attractors_t_windows = []

for ba, gs_percentage, i_e_w, i_freq in parameter_combinations:

    # --- simulation local vars ---------------------------------------------------------

    plastic_syn = False
    plastic_ux = True

    cue_percentage = 100
    
    # --- network initialization --------------------------------------------------------

    rcn = RecurrentCompetitiveNet(
        plasticity_rule = plasticity_rule,
        parameter_set = parameter_set)

    rcn.w_e_i = 3 * mV
    rcn.w_e_i = 3 * mV                     
    rcn.w_i_e = i_e_w * mV
    rcn.spont_rate = ba * Hz
    
    rcn.tau_Vth_e = 0.1 * second        
    rcn.k = 4                           

    rcn.net_init()

    rcn.w_max = 10 * mV                  

    rcn.set_stimulus_i(
        stimulus = 'flat_to_I', 
        frequency = i_freq * Hz)

    # --- set attractors ----------------------------------------------------------------

    attractors = []

    nIDs_attra = rcn.set_active_E_ids(stimulus = 'flat_to_E_fixed_size', offset = 0)
    rcn.set_potentiated_synapses(nIDs_attra)
    A1 = list(range(0, 64))
    attractors.append(('A1', A1))

    rcn.set_E_E_plastic(plastic = plastic_syn)
    rcn.set_E_E_ux_vars_plastic(plastic = plastic_ux)

    # --- run network -------------------------------------------------------------------

    # cue attractor 1

    act_ids = rcn.generic_stimulus(
        frequency = rcn.stim_freq_e, 
        stim_perc = cue_percentage, 
        subset = nIDs_attra)
    
    rcn.run_net(duration = 0.1)
   
    rcn.generic_stimulus_off(act_ids)

    rcn.run_net(duration = 10)

    # --- plotting simulation data ------------------------------------------------------

    # ps1stEnd, ps2ndStart = find_gap_between_PS(rcn, [('A1', nIDs_attra)], 0.75)

    visualize_attractor(attractor = nIDs_attra, network = rcn)
