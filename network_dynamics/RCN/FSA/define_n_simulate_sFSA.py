# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System / Artificial Intelligence
"""

import os, sys
import argparse
from brian2 import prefs

if sys.platform in ['linux', 'win32']:
    root = os.path.dirname(os.path.abspath(os.path.join(__file__ , '../../..')))
    sys.path.append(os.path.join(root, 'helper_functions', 'sFSA'))
    from sFSA import sFSA
else:
    from helper_functions.sFSA import sFSA

parser = argparse.ArgumentParser(description = 'sFSA')

parser.add_argument('--ba_rate', type = float,      default = 15.0,  help = 'Background mean rate (Hz).')
parser.add_argument('--W_ie', type = float,         default = 11.5,   help = 'I-to-E inhibition weight.')
parser.add_argument('--inh_rate', type = float,     default = 20.00, help = 'Inhibition mean rate (Hz).')
parser.add_argument('--w_acpt', type = float,       default = 1.6,  help = 'Weight in synapses to GO state (mV).')
parser.add_argument('--w_trans', type = float,      default = 24.00, help = 'Cue transfer weight (mV).')
parser.add_argument('--thr_GO_state', type = float, default = -48.5, help = 'Threshold for Vth gated synapses (mV).')
parser.add_argument('--gs_percent', type = float,   default = 100,   help = 'Percentage of stimulated neurons in attractor.')

args = parser.parse_args()

fsa = {
    'S': ['A', 'B'],                       # 1st state taken as 'start' state.
    'I': ['0', '1'],
    'T': ['(A, 0)->B', '(A, 1)->A', '(B, 0)->A', '(B, 1)->B']
}

sFSA_model = sFSA(fsa, args)                # create sFSA.

sFSA_model.feedInputWord(['0', '1'])        # feed input tokes (sequential).

data_path = sFSA_model.exportSfsaData()     # create simulaiton folder and export data.

sFSA_model.plotSfsaNetwork(data_path)       # plot RCN activity during token processing.