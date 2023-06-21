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

if sys.platform in ['linux', 'win32']:

    root = os.path.dirname(os.path.abspath(os.path.join(__file__ , '../../..')))

    sys.path.append(os.path.join(root, 'helper_functions'))
    sys.path.append(os.path.join(root, 'plotting_functions'))

    from recurrent_competitive_network import RecurrentCompetitiveNet
    from other import *
    from x_u_spks_from_basin import plot_x_u_spks_from_basin
    from rcn_spiketrains_histograms import plot_rcn_spiketrains_histograms
    # from spike_synchronisation import *
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

def gaussian_search(args):

    sigma = 3.0

    args.ba_rate = np.round(np.random.normal(args.ba_rate, sigma, 1)[0], 3)
    args.W_ie = np.round(np.random.normal(args.W_ie, sigma, 1)[0], 3)
    args.inh_rate = np.round(np.random.normal(args.inh_rate, sigma, 1)[0], 3)
    args.w_acpt = np.round(np.random.normal(args.w_acpt, sigma, 1)[0], 3)
    args.w_trans = np.round(np.random.normal(args.w_trans, sigma, 1)[0], 3)

# === parsing arguments =================================================================

_input_sequence = ['I']

parser = argparse.ArgumentParser(description='RNN_sFSA')

parser.add_argument('--ba_rate', type=float, default=15, help='Background mean rate (Hz).')
parser.add_argument('--gs_percent', type=float, default=100, help='Percentage of stimulated neurons in attractor.')
parser.add_argument('--W_ie', type=float, default=7, help='I-to-E inhibition weight.')
parser.add_argument('--inh_rate', type=float, default=20, help='Inhibition mean rate (Hz).')
parser.add_argument('--w_acpt', type=float, default=2.725,help='Weight in synapses to GO state (mV).')
parser.add_argument('--w_trans', type=float, default=22, help='Attractor state transition weight (mV).')
parser.add_argument('--thr_GO_state', type=float, default=-48.5, help='Threshold for Vth gated synapses (mV).')
parser.add_argument('--gauss_search', type=int, default=0, help='Gaussian search of parameters.')

args = parser.parse_args()

if args.gauss_search:

    gaussian_search(args)

# === local functions =================================================================

def save_config(args, path):

    with open(os.path.join(path, 'parameters.txt'), "w") as file:
        for arg in vars(args):
            arg_value = getattr(args, arg)
            if arg_value is not None:
                file.write(f"\n{arg}: {arg_value}\n")

def set_sFSA(rcn, args, stimulus_size = 64):

    # --- initializing some network parameters ---

    rcn.thr_GO_state = args.thr_GO_state
    rcn.E_E_syn_matrix_snapshot = False
    rcn.w_e_i = 3 * mV                      
    rcn.w_max = 10 * mV                     
    rcn.spont_rate = args.ba_rate * Hz
    rcn.w_e_i = 3 * mV                      
    rcn.w_i_e = args.W_ie * mV              
    rcn.tau_Vth_e = 0.1 * second            
    rcn.k = 4    
    rcn.U = 0.2

    # --- defining FSA as dictionary ---

    if print_report:

        print('\nDefining sFSA...\n')

    sFSA_state_dictionary = {'S': {}, 'I': {}, 'T': {}, 'N_e': 0, 'N_i': 0}

    list_S = []
    list_I = []
    _T = []

    # _S = input('> states (comma separated): ').split(', ')
    # _I = input('> inputs (comma separated): ').split(', ')

    _S = ['I', 'II', 'III']
    _I = ['x', 'y', 'z', 'u']
    _T = ['(I, x)->II', '(II, y)->I', '(I, z)->III', '(III, u)->II']

    # _ = input(f'> transition (type \'end\' to stop): ')

    # while _ != 'end':

    #     _T.append(_)

    #     _ = input(f'> transition (type \'end\' to stop): ')

    sFSA_state_dictionary['N_e'] = int((len(_S) + len(_I))*stimulus_size)
    sFSA_state_dictionary['N_i'] = int((sFSA_state_dictionary['N_e']*25)/100)

    _aux = 0
    _auxa = stimulus_size

    for s in _S:

        sFSA_state_dictionary['S'][s] = list(range(_aux, _auxa))

        _aux += stimulus_size
        _auxa += stimulus_size

    for i in _I:

        sFSA_state_dictionary['I'][i] = list(range(_aux, _auxa))

        _aux += stimulus_size
        _auxa += stimulus_size

    _aux = 0

    for t in _T:

        sFSA_state_dictionary['T'][_aux] = t

        _aux += 1

    # if print_report:

    #     for key, val in sFSA_state_dictionary.items():

    #         print(key, val)

    # --- initializing network ---

    rcn.N_input_e = sFSA_state_dictionary['N_e']
    rcn.N_input_i = sFSA_state_dictionary['N_i']

    rcn.stim_size_e = sFSA_state_dictionary['N_i']
    rcn.stim_size_i = sFSA_state_dictionary['N_i']

    rcn.N_e = sFSA_state_dictionary['N_e']
    rcn.N_i = sFSA_state_dictionary['N_i']

    rcn.net_init()

    rcn.set_stimulus_i(stimulus = 'flat_to_I', frequency = args.inh_rate * Hz)
    rcn.set_E_E_plastic(plastic = False)
    rcn.set_E_E_ux_vars_plastic(plastic = True)

    # --- setting attractros (tokens) ---

    if print_report:

        print('\nConfiguring RNN...\n')

    for parameter, values in sFSA_state_dictionary.items():

        if parameter in ['S', 'I']:

            for token, neur_IDs in values.items():

                rcn.set_potentiated_synapses(neur_IDs)

                if print_report:

                    print(f'set {token}')


    # --- setting state transitions ---

    if print_report:

        print('\nConfiguring state transitions...\n')

    for t_id, t in sFSA_state_dictionary['T'].items():

        set_state_transition(t, rcn, sFSA_state_dictionary)

        if print_report:

            print(f'transition {t_id}: {t}')

    return sFSA_state_dictionary

def set_state_transition(transition, rcn, sFSA):

    _x = transition.split('->')[0].split(', ')[0].replace('(', '')
    _i = transition.split('->')[0].split(', ')[1].replace(')', '')
    _y = transition.split('->')[1]

    # link input _i to current state _x
    rcn.set_synapses_A_2_B(A_ids = sFSA['I'][_i], B_ids = sFSA['S'][_x], weight = args.w_trans*mV)

    # set transition to _y
    rcn.set_synapses_A_2_GO(A_ids = sFSA['I'][_i], GO_ids = sFSA['S'][_y], weight = args.w_acpt*mV)
    rcn.set_synapses_A_2_GO(A_ids = sFSA['S'][_x], GO_ids = sFSA['S'][_y], weight = args.w_acpt*mV)

def feed_input_sequece(sequence, rcn, sFSA):

    _t0 = 0.2
    _t1 = 0.4
    _t2 = 0.6
    _t3 = 3.4
    _t4 = 1.0

    _input_tokens_twin = []

    def find_stimulus_ids(pattern, sFSA):
        if pattern in [key for key, val in sFSA['S'].items()]:
            return sFSA['S'][pattern], 'S'
        elif pattern in [key for key, val in sFSA['I'].items()]:
            return sFSA['I'][pattern], 'I'
        else:
            exit(f'\n[EROOR] patternd {pattern} not encoded in RNN (exiting...).\n')

    _input = sequence

    if print_report:

        print(f'\ninputing sequence: {_input}...\n')

    for a in range(len(_input)):

        print(f'> {_input[a]}')

        _stim, _class = find_stimulus_ids(_input[a], sFSA) # class is either state or input token

        act_ids = rcn.generic_stimulus(
                frequency = rcn.stim_freq_e, 
                stim_perc = args.gs_percent, 
                subset = _stim)

        # cue attractor

        if a == 0:

            rcn.run_net(duration = _t0)

            rcn.generic_stimulus_off(act_ids)

            if _class == 'I':

                _input_tokens_twin.append(((rcn.E_rec.t[-1]/second)-0.2, rcn.E_rec.t[-1]/second))

            # free network activity

            rcn.run_net(duration = _t1)

        else:

            rcn.run_net(duration = _t2)

            if _class == 'I':

                _input_tokens_twin.append(((rcn.E_rec.t[-1]/second)-0.7, rcn.E_rec.t[-1]/second))

            rcn.generic_stimulus_off(act_ids)

            # free network activity

            rcn.run_net(duration = _t3)

    rcn.run_net(duration = _t4)

    return _input_tokens_twin, [_t0, _t1, _t2, _t3, _t4]


def export_attractors_data(rcn, sFSA, _input_sequence, _input_tokens_twin = [], stimulation_twin = []):

    # Excitatory spike trains.

    _E_spk_trains = {key: value for key, value in rcn.E_mon.spike_trains().items()}

    # Sim t.

    sim_t = rcn.E_rec.t/second

    # Export.

    fn = os.path.join(
        rcn.net_sim_data_path,
        'sSFA_sim_data.pickle')

    with open(fn, 'wb') as f:
        pickle.dump({
            'E_spk_trains': _E_spk_trains,
            'sim_t': sim_t,
            'sFSA': sFSA,
            'i_tokens_twindows': _input_tokens_twin,
            'input_sequence': _input_sequence,
            'stimulation_time_windows': stimulation_twin
        }, f)


# === simulation parameters =============================================================

if print_report:

    print('''\n
        This script instanciate and simulates our spiking Finite-State Automata (sFSA)
    You\'ll be asked to input a list S of states, a list I of input symbols and a list T
    of transitions in the form (x, i)->y with x, y in S and i in I.''')

timestamp_folder = make_timestamped_folder('../../../results/RCN_FSA/')

plasticity_rule = 'LR4'
parameter_set = '2.2'

# === initializing/running network ======================================================
    
# --initialise time to predict overall experiment time

rcn = RecurrentCompetitiveNet(
    plasticity_rule = plasticity_rule,
    parameter_set = parameter_set)

sFSA = set_sFSA(rcn, args)

save_dir = os.path.join(os.path.join(timestamp_folder, f'BA_{args.ba_rate}_GS_{args.gs_percent}_W_{args.W_ie}_I_{args.inh_rate}'))
os.mkdir(save_dir)
rcn.net_sim_data_path = save_dir

save_config(args, rcn.net_sim_data_path)

_input_tokens_twin, stimulation_twin = feed_input_sequece(_input_sequence, rcn, sFSA)

# 2 ------ plotting simulation data ------

export_attractors_data(rcn, sFSA, _input_sequence, _input_tokens_twin, stimulation_twin)