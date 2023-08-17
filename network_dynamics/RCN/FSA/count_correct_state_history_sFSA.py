# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System / Artificial Intelligence
"""

import os, sys
import argparse
from brian2 import prefs
import numpy as np

if sys.platform in ['linux', 'win32']:
    root = os.path.dirname(os.path.abspath(os.path.join(__file__ , '../../..')))
    sys.path.append(os.path.join(root, 'helper_functions', 'sFSA'))
    from sFSA import sFSA
else:
    from helper_functions.sFSA import sFSA

parser = argparse.ArgumentParser(description = 'sFSA')

# General parameters.
parser.add_argument('--ba_rate', type = float,          default = 15.00,  help = 'Background mean rate (Hz).')
parser.add_argument('--e_e_max_weight', type = float,   default = 9.00,   help = 'Within attractor weight (mV).')
parser.add_argument('--W_ei', type = float,             default = 1.50,   help = 'E-to-I weight (mV).')
parser.add_argument('--W_ie', type = float,             default = 11.5,   help = 'I-to-E weight (mV).')
parser.add_argument('--inh_rate', type = float,         default = 20.00,  help = 'Inhibition mean rate (Hz).')
parser.add_argument('--gs_percent', type = float,       default = 100,    help = 'Percentage of stimulated neurons in attractor.')

# State transition specific parameters.
parser.add_argument('--w_acpt', type = float,           default = 2.0,   help = 'Weight in synapses to GO state (mV).')
parser.add_argument('--w_trans', type = float,          default = 7.8,   help = 'Cue transfer weight (mV).')
parser.add_argument('--thr_GO_state', type = float,     default = -47.00,  help = 'Threshold for Vth gated synapses (mV).')

args = parser.parse_args()

fsa = {
    'S': ['A'],
    'I': ['0'],
    'T': []
}

sFSA_model = sFSA(fsa, args)

correct_states = ['A', 'B']

t_folders = ['1', '2', '3', '4', '5', '6', '7', '8', 'A', 'B', 'C', 'I', 'II']


for target_foler in t_folders:

    sim_folders = os.listdir(os.path.join('D://A_PhD//GitHub//wm_colaboration//results//', target_foler))

    correct = 0
    null = 0
    multiple_active = 0
    no_trans = 0

    for folder in sim_folders:

        folder_path = os.path.join('D://A_PhD//GitHub//wm_colaboration//results//', target_foler, folder)

        state_history = sFSA_model.computeStateHistory(folder_path)

        if state_history == ['A', 'two']:
            multiple_active += 1
        elif state_history == ['A', 'null']:
            null += 1
        elif state_history == correct_states:
            correct += 1
        elif state_history == ['A', 'A']:
            no_trans += 1
        else:
            pass

    correct = np.round(correct/len(sim_folders)*100, 2)
    multiple_active = np.round(multiple_active/len(sim_folders)*100, 2)
    null = np.round(null/len(sim_folders)*100, 2)
    no_trans = np.round(no_trans/len(sim_folders)*100, 2)

    print(target_foler)
    print(f'correct: {correct}%, multiple active: {multiple_active}%, null: {null}%, no trans.: {no_trans}%\n')