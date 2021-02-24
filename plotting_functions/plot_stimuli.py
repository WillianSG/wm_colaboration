# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
- Plot stimuli presented to the network.
"""
import os, sys
from numpy import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from brian2 import *

helper_dir = 'helper_functions'

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))

simulation_folder = os.path.join(parent_dir, 'net_simulation')

if not(os.path.isdir(simulation_folder)):
	os.mkdir(simulation_folder)

detination_dir = os.path.join(simulation_folder, 'stimuli_plots')

if not(os.path.isdir(detination_dir)):
	os.mkdir(detination_dir)

# Helper modules
from att_net_obj import AttractorNetwork

stimuli = ['square', 'circle', 'triangle', 'cross', 'random', 'flat_to_I', 'flat_to_E', 'flat_to_E_fixed_size']

# Starts a new scope for magic functions
start_scope()

n = AttractorNetwork()

n.t_run = 1*second

n.int_meth_neur = 'linear'
n.int_meth_syn = 'euler'
n.stim_type_e = 'flat_to_E'

n.stim_size_e = 64
n.stim_freq_e = 3900*Hz
n.stim_offset_e = 96

n.stim_type_i = 'flat_to_I'
n.stim_size_i = n.N_i
n.stim_freq_i = 5000*Hz

# Spikemonitors
n.Input_to_E_mon_record = True
n.Input_to_I_mon_record = False
n.E_mon_record = False
n.I_mon_record = False

# @network_operation settings

# Rho snapshots
n.rho_matrix_snapshots = False
n.rho_matrix_snapshots_step = 1000 *ms

# Weight snapshots
n.w_matrix_snapshots = False
n.w_matrix_snapshots_step = 1000*ms 

# Xpre snapshots
n.xpre_matrix_snapshots = False
n.xpre_matrix_snapshots_step = 1000*ms

# Xpost snapshots
n.xpost_matrix_snapshots = False
n.xpost_matrix_snapshots_step = 1000*ms

# variables for @network_operation function for stimulus change
n.stimulus_pulse = True
n.stimulus_pulse_clock_dt = 3*second

n.stim_freq_original = n.stim_freq_e

# Learning rule 
n.plasticity_rule = 'LR2' # LR1, LR2
n.parameter = '2.4' #
n.neuron_type = 'LIF' #
n.bistability = True

# Connection weights
n.w_input_e = 1*mV
n.w_input_i = 1*mV
n.w_e_e = 0.5*mV
n.w_e_i = 1*mV        
n.w_i_e = 1*mV

# Other connection weight variables
n.w_e_e_max = 7.25*mV # max weight in EE connections

n.fixed_attractor = False # [?]
n.fixed_attractor_wmax = 'all_max' # [?]

# Set pulse duration
n.stimulus_pulse_duration = n.stimulus_pulse_clock_dt

# Name of simulation
n.exp_type = 'stimuli_plots'

n.init_network_modules() # creates/initializes all objects/parameters/mons

n.run_net()

plt.plot(n.Input_to_E_mon.t/ms, n.Input_to_E_mon.i, '.k')
plt.title(n.stim_type_e)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.savefig(os.path.join(detination_dir, n.stim_type_e + '.png'))
plt.close('all')




