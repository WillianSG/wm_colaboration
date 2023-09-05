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


def compute_transition(current_state, input_symbol, fsa):
    '''
    Function parsing bin. number to extract correct state transitions.
    '''

    match = f'({current_state}, {input_symbol})'

    for transition in fsa['T']:

        if transition.find(match) != -1:
            return transition.split('->')[-1]


def run_sfsa(params, tmp_folder=".", word_length=4, save_plot=None, seed_init=None, record_traces=False):
    '''
    Configures a RCN to implement a FSA to recognize a random binary word.
    '''

    pid = os.getpid()

    # - Define FSA (1st state in S taken as 'start' state).

    fsa = {
        'S': ['A', 'B'],  # states
        'I': ['0', '1'],  # symbols
        'T': ['(A, 0)->B', '(A, 1)->A', '(B, 0)->A', '(B, 1)->B']  # transitions
    }

    # - Create input word.

    binary_word = np.random.randint(2, size=word_length)
    binary_word = [str(dig) for dig in binary_word]

    true_state_transitions = [fsa['S'][0]]  # this FSM starts at 'A'

    for dig in binary_word:  # computing true state transitions
        true_state_transitions.append(compute_transition(true_state_transitions[-1], dig, fsa))

    # - Create sFSA.

    _rcn_path = make_timestamped_folder(os.path.join(tmp_folder, f"{pid}"))

    sFSA_model = sFSA(FSA_model = fsa, params = params, RCN_path = _rcn_path, seed_init = seed_init, record_traces = record_traces)

    # - Folder for simulation results.

    sFSA_model.setSimulationFolder(_rcn_path)  # path
    _f = os.path.join(sFSA_model.data_folder, "simulation_summary.txt")  # output .txt with simulation summary

    # - Feed input word.

    sFSA_model.feedInputWord(binary_word)

    sFSA_model.exportSfsaData(network_plot=save_plot, pickle_dump=False)  # computes state transisitons

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
