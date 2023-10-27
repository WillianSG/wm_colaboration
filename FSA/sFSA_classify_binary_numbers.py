# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System
"""

import os, sys, pickle
import argparse
from brian2 import prefs

if sys.platform in ['linux', 'win32']:
    root = os.path.dirname(os.path.abspath(os.path.join(__file__, '../../..')))
    sys.path.append(os.path.join(root, 'helper_functions'))
    from sFSA import sFSA
else:
    from helper_functions.sFSA import sFSA

parser = argparse.ArgumentParser(description='sFSA')

parser.add_argument('--background_activity', type=float, default=15.00, help='Background mean rate (Hz).')
parser.add_argument('--e_e_max_weight', type=float, default=9.0, help='Within attractor weight (mV).')
parser.add_argument('--e_i_weight', type=float, default=2.1, help='E-to-I weight (mV).')
parser.add_argument('--i_e_weight', type=float, default=3.5, help='I-to-E weight (mV).')
parser.add_argument('--i_freq', type=float, default=20.00, help='Inhibition mean rate (Hz).')
parser.add_argument('--cue_percentage', type=float, default=100, help='Percentage of stimulated neurons in attractor.')
parser.add_argument('--w_acpt', type=float, default=2.55, help='Weight in synapses to GO state (mV).')
parser.add_argument('--w_trans', type=float, default=7.8, help='Cue transfer weight (mV).')
parser.add_argument('--thr_GO_state', type=float, default=-47.5, help='Threshold for Vth gated synapses (mV).')
parser.add_argument('--delay_A2GO', type=float, default=1.5, help='Connection delay of red synapses (s).')
parser.add_argument('--delay_gap_A2B', type=float, default=0.0, help='Connection delay of blue synapses (s).')
parser.add_argument('--cue_length', type=float, default=0.6, help='Time over which an attractor is stimulated (s).')

args = parser.parse_args()

args.seed_init = 0
args.record_traces = True

params = {}

for arg in vars(args):
    params[arg] = getattr(args, arg)

params = {'background_activity': 13.855558143033827, 'cue_length': 0.513141209303625, 'cue_percentage': 100,
          'delay_A2GO': 2.0506400454172526, 'delay_gap_A2B': 0.06729242159249718, 'e_e_max_weight': 14.337885250035841,
          'e_i_weight': 4.253159333783978, 'i_e_weight': 5.302236785709991, 'i_frequency': 31.841289668071244,
          'thr_GO_state': -52.56353267076404, 'w_acpt': 1.4391073837024586, 'w_trans': 6.898596074987625}

# 1st state in S taken as 'start' state.
fsa = {
    'S': ['A', 'B'],
    'I': ['0', '1'],
    'T': ['(A, 0)->B', '(A, 1)->A', '(B, 0)->A', '(B, 1)->B']
}

# create sFSA.

sFSA_model = sFSA(fsa, params, RCN_path='')

sFSA_model.makeSimulationFolder()

sFSA_model.storeSfsa()

# classifying digits.

digits = [['0', '1'], ['0', '1', '0'], ['1', '1'], ['0', '0'], ['1', '0', '1'], ['1', '0']]
true_state_hist = [[fsa['S'][0], 'B', 'B'], ['A', 'B', 'B', 'A'], ['A', 'A', 'A'], ['A', 'B', 'A'],
                   ['A', 'A', 'B', 'B'], ['A', 'A', 'B']]

correct = 0
null = 0
double_activation = 0

for i in range(len(digits)):

    sFSA_model.restoreSfsa()

    sFSA_model.feedInputWord(digits[i])  # feed input tokes.

    sFSA_model.exportSfsaData(network_plot=True, pickle_dump=False,
                              name_ext=f'_{i}')  # create simulaiton folder and export data.

    state_hist = sFSA_model.sFSA_sim_data['state_history']

    if state_hist[0] == 'two':
        state_hist[0] = 'A'

    if state_hist == true_state_hist[i]:
        correct += 1

    if 'null' in state_hist:
        null += 1

    if 'two' in state_hist:
        double_activation += 1

    accuracy = sFSA_model.computeAccuracy(true_state_hist, state_hist)

    print(f'true_state_hist: {true_state_hist[i]}, pred. state_hist: {state_hist}')
    print(f'accuracy: {accuracy}')

CR = correct / len(digits)

print(f'CR: {CR}')

# exporting data.

with open(os.path.join(sFSA_model.data_folder, 'performance.txt'), "w") as file:
    file.write(f'\ndigits count           : {len(digits)}')
    file.write(f'\nCR                     : {CR}')
    file.write(f'\ncorrect count          : {correct}')
    file.write(f'\nnull count             : {null}')
    file.write(f'\ndouble activation count: {double_activation}')
