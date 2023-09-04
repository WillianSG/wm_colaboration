# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System
"""

import shutil, os, argparse
import numpy as np

from helper_functions.other import make_timestamped_folder
from helper_functions.sFSA import sFSA

# - Function parsing bin. number to extract correct state transitions.
def compute_transition(current_state, input_symbol, fsa):

    match = f'({current_state}, {input_symbol})'

    for transition in fsa['T']:

        if transition.find(match) != -1:
            return transition.split('->')[-1]

def run_sfsa(params, tmp_folder = ".", word_length = 4, save_plot = False, seed_init = None):

    pid = os.getpid()

    # - Parameters to configure sFSA are passed as argparser.

    parser = argparse.ArgumentParser(description = 'sFSA')

    parser.add_argument('--ba_rate', type = float,          default = 15.00,  help = 'Background mean rate (Hz).')
    parser.add_argument('--e_e_max_weight', type = float,   default = 10.0,   help = 'Within attractor weight (mV).')
    parser.add_argument('--W_ei', type = float,             default = 2.1,   help = 'E-to-I weight (mV).')
    parser.add_argument('--W_ie', type = float,             default = 11.5,   help = 'I-to-E weight (mV).')
    parser.add_argument('--inh_rate', type = float,         default = 20.00,  help = 'Inhibition mean rate (Hz).')
    parser.add_argument('--gs_percent', type = float,       default = 100,    help = 'Percentage of stimulated neurons in attractor.')
    parser.add_argument('--w_acpt', type = float,           default = 2.55,   help = 'Weight in synapses to GO state (mV).')
    parser.add_argument('--w_trans', type = float,          default = 7.8,   help = 'Cue transfer weight (mV).')
    parser.add_argument('--thr_GO_state', type = float,     default = -48.0,  help = 'Threshold for Vth gated synapses (mV).')
    parser.add_argument('--delay_A2GO', type = float,     default = 2.5,  help = 'Connection delay of red synapses (s).')
    parser.add_argument('--delay_A2B', type = float,     default = 0.63,  help = 'Connection delay of blue synapses (s).')

    args = parser.parse_args()

    args.ba_rate = params['background_activity']
    args.e_e_max_weight = params['e_e_max_weight']
    args.W_ei = params['e_i_weight']
    args.W_ie = params['i_e_weight']
    args.inh_rate = params['i_freq']
    args.gs_percent = params['cue_percentage']
    args.w_acpt = params['w_acpt']
    args.w_trans = params['w_trans']
    args.thr_GO_state = params['thr_GO_state']
    args.delay_A2GO = params['delay_A2GO']
    args.delay_A2B = params['delay_A2B']

    args.seed_init = seed_init

    # - Define FSA (1st state in S taken as 'start' state).

    fsa = {
        'S': ['A', 'B'],                                                # states
        'I': ['0', '1'],                                                # symbols
        'T': ['(A, 0)->B', '(A, 1)->A', '(B, 0)->A', '(B, 1)->B']       # transitions
    }

    # - Create input word.

    binary_word = np.random.randint(2, size = word_length)

    true_state_transitions = ['A']                                      # this FSM starts at 'A'
    
    for dig in binary_word:                                             # computing true state transitions
        true_state_transitions.append(compute_transition(true_state_transitions[-1], dig, fsa))

    # - Create sFSA.

    sFSA_model = sFSA(fsa, args)

    # - Folder for simulation results.

    sFSA_model.setSimulationFolder(make_timestamped_folder(os.path.join(tmp_folder, f"{pid}")))  # path
    _f = os.path.join(sFSA_model.data_folder, "simulation_summary.txt")                          # output .txt with simulation summary

    # - Feed input word.

    sFSA_model.feedInputWord(binary_word)

    sFSA_model.exportSfsaData(network_plot = save_plot, pickle_dump = save_plot)                     # computes state transisitons

    pred_state_transitions = sFSA_model.sFSA_sim_data['state_history']

    f1_score, recall, precision = sFSA_model.computeF1Score(true_state_transitions, pred_state_transitions)

    # - Cleanup.

    shutil.rmtree(sFSA_model.data_folder)
    
    del sFSA_model

    params.pop("worker_id", None)

    return params | {
        "f1_score": f1_score,
        "recall": recall,
        "accuracy": precision
    }