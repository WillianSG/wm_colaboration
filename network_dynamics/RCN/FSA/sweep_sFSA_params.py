# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System
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

parser.add_argument('--params_set', type = str, help = 'Parameter set used for simulation.')

arg_init = parser.parse_args()

if arg_init.params_set == '1':
    # param. set 1

    parser.add_argument('--ba_rate', type = float,          default = 15.00,  help = 'Background mean rate (Hz).')
    parser.add_argument('--e_e_max_weight', type = float,   default = 9.0,   help = 'Within attractor weight (mV).')
    parser.add_argument('--W_ei', type = float,             default = 1.75,   help = 'E-to-I weight (mV).')
    parser.add_argument('--W_ie', type = float,             default = 11.5,   help = 'I-to-E weight (mV).')
    parser.add_argument('--inh_rate', type = float,         default = 20.00,  help = 'Inhibition mean rate (Hz).')
    parser.add_argument('--gs_percent', type = float,       default = 100,    help = 'Percentage of stimulated neurons in attractor.')
    parser.add_argument('--w_acpt', type = float,           default = 2.25,   help = 'Weight in synapses to GO state (mV).')
    parser.add_argument('--w_trans', type = float,          default = 7.8,   help = 'Cue transfer weight (mV).')
    parser.add_argument('--thr_GO_state', type = float,     default = -48.5,  help = 'Threshold for Vth gated synapses (mV).')
    parser.add_argument('--delay_A2GO', type = float,     default = 2,  help = 'Connection delay of red synapses (s).')
    parser.add_argument('--delay_A2B', type = float,     default = 0.625,  help = 'Connection delay of blue synapses (s).')

if arg_init.params_set == '2':
    # param. set 2

    parser.add_argument('--ba_rate', type = float,          default = 15.00,  help = 'Background mean rate (Hz).')
    parser.add_argument('--e_e_max_weight', type = float,   default = 9.0,   help = 'Within attractor weight (mV).')
    parser.add_argument('--W_ei', type = float,             default = 1.85,   help = 'E-to-I weight (mV).')
    parser.add_argument('--W_ie', type = float,             default = 11.5,   help = 'I-to-E weight (mV).')
    parser.add_argument('--inh_rate', type = float,         default = 20.00,  help = 'Inhibition mean rate (Hz).')
    parser.add_argument('--gs_percent', type = float,       default = 100,    help = 'Percentage of stimulated neurons in attractor.')
    parser.add_argument('--w_acpt', type = float,           default = 2.25,   help = 'Weight in synapses to GO state (mV).')
    parser.add_argument('--w_trans', type = float,          default = 7.8,   help = 'Cue transfer weight (mV).')
    parser.add_argument('--thr_GO_state', type = float,     default = -48.5,  help = 'Threshold for Vth gated synapses (mV).')
    parser.add_argument('--delay_A2GO', type = float,     default = 2,  help = 'Connection delay of red synapses (s).')
    parser.add_argument('--delay_A2B', type = float,     default = 0.625,  help = 'Connection delay of blue synapses (s).')

if arg_init.params_set == '4':
    # param. set 4

    parser.add_argument('--ba_rate', type = float,          default = 15.00,  help = 'Background mean rate (Hz).')
    parser.add_argument('--e_e_max_weight', type = float,   default = 9.0,   help = 'Within attractor weight (mV).')
    parser.add_argument('--W_ei', type = float,             default = 2.1,   help = 'E-to-I weight (mV).')
    parser.add_argument('--W_ie', type = float,             default = 11.5,   help = 'I-to-E weight (mV).')
    parser.add_argument('--inh_rate', type = float,         default = 20.00,  help = 'Inhibition mean rate (Hz).')
    parser.add_argument('--gs_percent', type = float,       default = 100,    help = 'Percentage of stimulated neurons in attractor.')
    parser.add_argument('--w_acpt', type = float,           default = 2.43,   help = 'Weight in synapses to GO state (mV).')
    parser.add_argument('--w_trans', type = float,          default = 7.8,   help = 'Cue transfer weight (mV).')
    parser.add_argument('--thr_GO_state', type = float,     default = -48.0,  help = 'Threshold for Vth gated synapses (mV).')
    parser.add_argument('--delay_A2GO', type = float,     default = 2,  help = 'Connection delay of red synapses (s).')
    parser.add_argument('--delay_A2B', type = float,     default = 0.625,  help = 'Connection delay of blue synapses (s).')

if arg_init.params_set == '5':
    # param. set 5

    parser.add_argument('--ba_rate', type = float,          default = 15.00,  help = 'Background mean rate (Hz).')
    parser.add_argument('--e_e_max_weight', type = float,   default = 9.0,   help = 'Within attractor weight (mV).')
    parser.add_argument('--W_ei', type = float,             default = 2.1,   help = 'E-to-I weight (mV).')
    parser.add_argument('--W_ie', type = float,             default = 11.5,   help = 'I-to-E weight (mV).')
    parser.add_argument('--inh_rate', type = float,         default = 20.00,  help = 'Inhibition mean rate (Hz).')
    parser.add_argument('--gs_percent', type = float,       default = 100,    help = 'Percentage of stimulated neurons in attractor.')
    parser.add_argument('--w_acpt', type = float,           default = 2.51,   help = 'Weight in synapses to GO state (mV).')
    parser.add_argument('--w_trans', type = float,          default = 7.8,   help = 'Cue transfer weight (mV).')
    parser.add_argument('--thr_GO_state', type = float,     default = -48.0,  help = 'Threshold for Vth gated synapses (mV).')
    parser.add_argument('--delay_A2GO', type = float,     default = 2,  help = 'Connection delay of red synapses (s).')
    parser.add_argument('--delay_A2B', type = float,     default = 0.625,  help = 'Connection delay of blue synapses (s).')

if arg_init.params_set == '7':
    # param. set 7

    parser.add_argument('--ba_rate', type = float,          default = 15.00,  help = 'Background mean rate (Hz).')
    parser.add_argument('--e_e_max_weight', type = float,   default = 9.0,   help = 'Within attractor weight (mV).')
    parser.add_argument('--W_ei', type = float,             default = 2.1,   help = 'E-to-I weight (mV).')
    parser.add_argument('--W_ie', type = float,             default = 11.5,   help = 'I-to-E weight (mV).')
    parser.add_argument('--inh_rate', type = float,         default = 20.00,  help = 'Inhibition mean rate (Hz).')
    parser.add_argument('--gs_percent', type = float,       default = 100,    help = 'Percentage of stimulated neurons in attractor.')
    parser.add_argument('--w_acpt', type = float,           default = 2.55,   help = 'Weight in synapses to GO state (mV).')
    parser.add_argument('--w_trans', type = float,          default = 7.8,   help = 'Cue transfer weight (mV).')
    parser.add_argument('--thr_GO_state', type = float,     default = -48.0,  help = 'Threshold for Vth gated synapses (mV).')
    parser.add_argument('--delay_A2GO', type = float,     default = 2,  help = 'Connection delay of red synapses (s).')
    parser.add_argument('--delay_A2B', type = float,     default = 0.625,  help = 'Connection delay of blue synapses (s).')

if arg_init.params_set == '8':
    # param. set 8

    parser.add_argument('--ba_rate', type = float,          default = 15.00,  help = 'Background mean rate (Hz).')
    parser.add_argument('--e_e_max_weight', type = float,   default = 8.25,   help = 'Within attractor weight (mV).')
    parser.add_argument('--W_ei', type = float,             default = 2.1,   help = 'E-to-I weight (mV).')
    parser.add_argument('--W_ie', type = float,             default = 11.5,   help = 'I-to-E weight (mV).')
    parser.add_argument('--inh_rate', type = float,         default = 20.00,  help = 'Inhibition mean rate (Hz).')
    parser.add_argument('--gs_percent', type = float,       default = 100,    help = 'Percentage of stimulated neurons in attractor.')
    parser.add_argument('--w_acpt', type = float,           default = 2.51,   help = 'Weight in synapses to GO state (mV).')
    parser.add_argument('--w_trans', type = float,          default = 7.8,   help = 'Cue transfer weight (mV).')
    parser.add_argument('--thr_GO_state', type = float,     default = -48.0,  help = 'Threshold for Vth gated synapses (mV).')
    parser.add_argument('--delay_A2GO', type = float,     default = 2,  help = 'Connection delay of red synapses (s).')
    parser.add_argument('--delay_A2B', type = float,     default = 0.525,  help = 'Connection delay of blue synapses (s).')

if arg_init.params_set == 'A':
    # param. set A

    parser.add_argument('--ba_rate', type = float,          default = 15.00,  help = 'Background mean rate (Hz).')
    parser.add_argument('--e_e_max_weight', type = float,   default = 9.0,   help = 'Within attractor weight (mV).')
    parser.add_argument('--W_ei', type = float,             default = 2.1,   help = 'E-to-I weight (mV).')
    parser.add_argument('--W_ie', type = float,             default = 11.5,   help = 'I-to-E weight (mV).')
    parser.add_argument('--inh_rate', type = float,         default = 20.00,  help = 'Inhibition mean rate (Hz).')
    parser.add_argument('--gs_percent', type = float,       default = 100,    help = 'Percentage of stimulated neurons in attractor.')
    parser.add_argument('--w_acpt', type = float,           default = 2.51,   help = 'Weight in synapses to GO state (mV).')
    parser.add_argument('--w_trans', type = float,          default = 7.8,   help = 'Cue transfer weight (mV).')
    parser.add_argument('--thr_GO_state', type = float,     default = -48.0,  help = 'Threshold for Vth gated synapses (mV).')
    parser.add_argument('--delay_A2GO', type = float,     default = 1.5,  help = 'Connection delay of red synapses (s).')
    parser.add_argument('--delay_A2B', type = float,     default = 0.625,  help = 'Connection delay of blue synapses (s).')

if arg_init.params_set == 'B':
    # param. set B

    parser.add_argument('--ba_rate', type = float,          default = 15.00,  help = 'Background mean rate (Hz).')
    parser.add_argument('--e_e_max_weight', type = float,   default = 9.0,   help = 'Within attractor weight (mV).')
    parser.add_argument('--W_ei', type = float,             default = 2.1,   help = 'E-to-I weight (mV).')
    parser.add_argument('--W_ie', type = float,             default = 11.5,   help = 'I-to-E weight (mV).')
    parser.add_argument('--inh_rate', type = float,         default = 20.00,  help = 'Inhibition mean rate (Hz).')
    parser.add_argument('--gs_percent', type = float,       default = 100,    help = 'Percentage of stimulated neurons in attractor.')
    parser.add_argument('--w_acpt', type = float,           default = 2.51,   help = 'Weight in synapses to GO state (mV).')
    parser.add_argument('--w_trans', type = float,          default = 7.8,   help = 'Cue transfer weight (mV).')
    parser.add_argument('--thr_GO_state', type = float,     default = -48.0,  help = 'Threshold for Vth gated synapses (mV).')
    parser.add_argument('--delay_A2GO', type = float,     default = 1.6,  help = 'Connection delay of red synapses (s).')
    parser.add_argument('--delay_A2B', type = float,     default = 0.625,  help = 'Connection delay of blue synapses (s).')

if arg_init.params_set == 'C':
    # param. set C

    parser.add_argument('--ba_rate', type = float,          default = 15.00,  help = 'Background mean rate (Hz).')
    parser.add_argument('--e_e_max_weight', type = float,   default = 9.0,   help = 'Within attractor weight (mV).')
    parser.add_argument('--W_ei', type = float,             default = 2.1,   help = 'E-to-I weight (mV).')
    parser.add_argument('--W_ie', type = float,             default = 11.5,   help = 'I-to-E weight (mV).')
    parser.add_argument('--inh_rate', type = float,         default = 20.00,  help = 'Inhibition mean rate (Hz).')
    parser.add_argument('--gs_percent', type = float,       default = 100,    help = 'Percentage of stimulated neurons in attractor.')
    parser.add_argument('--w_acpt', type = float,           default = 2.51,   help = 'Weight in synapses to GO state (mV).')
    parser.add_argument('--w_trans', type = float,          default = 7.8,   help = 'Cue transfer weight (mV).')
    parser.add_argument('--thr_GO_state', type = float,     default = -48.0,  help = 'Threshold for Vth gated synapses (mV).')
    parser.add_argument('--delay_A2GO', type = float,     default = 2.3,  help = 'Connection delay of red synapses (s).')
    parser.add_argument('--delay_A2B', type = float,     default = 0.625,  help = 'Connection delay of blue synapses (s).')

if arg_init.params_set == 'I':
    # param. set I

    parser.add_argument('--ba_rate', type = float,          default = 15.00,  help = 'Background mean rate (Hz).')
    parser.add_argument('--e_e_max_weight', type = float,   default = 9.0,   help = 'Within attractor weight (mV).')
    parser.add_argument('--W_ei', type = float,             default = 2.1,   help = 'E-to-I weight (mV).')
    parser.add_argument('--W_ie', type = float,             default = 11.5,   help = 'I-to-E weight (mV).')
    parser.add_argument('--inh_rate', type = float,         default = 20.00,  help = 'Inhibition mean rate (Hz).')
    parser.add_argument('--gs_percent', type = float,       default = 100,    help = 'Percentage of stimulated neurons in attractor.')
    parser.add_argument('--w_acpt', type = float,           default = 2.51,   help = 'Weight in synapses to GO state (mV).')
    parser.add_argument('--w_trans', type = float,          default = 7.8,   help = 'Cue transfer weight (mV).')
    parser.add_argument('--thr_GO_state', type = float,     default = -48.0,  help = 'Threshold for Vth gated synapses (mV).')
    parser.add_argument('--delay_A2GO', type = float,     default = 2,  help = 'Connection delay of red synapses (s).')
    parser.add_argument('--delay_A2B', type = float,     default = 0.525,  help = 'Connection delay of blue synapses (s).')

if arg_init.params_set == 'II':
    # param. set II

    parser.add_argument('--ba_rate', type = float,          default = 15.00,  help = 'Background mean rate (Hz).')
    parser.add_argument('--e_e_max_weight', type = float,   default = 9.0,   help = 'Within attractor weight (mV).')
    parser.add_argument('--W_ei', type = float,             default = 2.1,   help = 'E-to-I weight (mV).')
    parser.add_argument('--W_ie', type = float,             default = 11.5,   help = 'I-to-E weight (mV).')
    parser.add_argument('--inh_rate', type = float,         default = 20.00,  help = 'Inhibition mean rate (Hz).')
    parser.add_argument('--gs_percent', type = float,       default = 100,    help = 'Percentage of stimulated neurons in attractor.')
    parser.add_argument('--w_acpt', type = float,           default = 2.51,   help = 'Weight in synapses to GO state (mV).')
    parser.add_argument('--w_trans', type = float,          default = 7.8,   help = 'Cue transfer weight (mV).')
    parser.add_argument('--thr_GO_state', type = float,     default = -48.0,  help = 'Threshold for Vth gated synapses (mV).')
    parser.add_argument('--delay_A2GO', type = float,     default = 2,  help = 'Connection delay of red synapses (s).')
    parser.add_argument('--delay_A2B', type = float,     default = 0.45,  help = 'Connection delay of blue synapses (s).')

if arg_init.params_set == 'III':
    # param. set III

    parser.add_argument('--ba_rate', type = float,          default = 15.00,  help = 'Background mean rate (Hz).')
    parser.add_argument('--e_e_max_weight', type = float,   default = 9.0,   help = 'Within attractor weight (mV).')
    parser.add_argument('--W_ei', type = float,             default = 2.1,   help = 'E-to-I weight (mV).')
    parser.add_argument('--W_ie', type = float,             default = 11.5,   help = 'I-to-E weight (mV).')
    parser.add_argument('--inh_rate', type = float,         default = 20.00,  help = 'Inhibition mean rate (Hz).')
    parser.add_argument('--gs_percent', type = float,       default = 100,    help = 'Percentage of stimulated neurons in attractor.')
    parser.add_argument('--w_acpt', type = float,           default = 2.55,   help = 'Weight in synapses to GO state (mV).')
    parser.add_argument('--w_trans', type = float,          default = 7.8,   help = 'Cue transfer weight (mV).')
    parser.add_argument('--thr_GO_state', type = float,     default = -48.0,  help = 'Threshold for Vth gated synapses (mV).')
    parser.add_argument('--delay_A2GO', type = float,     default = 2,  help = 'Connection delay of red synapses (s).')
    parser.add_argument('--delay_A2B', type = float,     default = 0.525,  help = 'Connection delay of blue synapses (s).')

args = parser.parse_args()

# 1st state in S taken as 'start' state.
fsa = {
    'S': ['A', 'B'],
    'I': ['0', '1'],
    'T': ['(A, 0)->B', '(A, 1)->A', '(B, 0)->A', '(B, 1)->B']
}

# create sFSA.
sFSA_model = sFSA(fsa, args)

# feed input tokes.
sFSA_model.feedInputWord(['0'])

# create simulaiton folder and export data.
sFSA_model.exportSfsaData(sub_dir = args.params_set)