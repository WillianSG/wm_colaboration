# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
- Learning of stimulus in an attractor network.
- https://brian2.readthedocs.io/en/stable/user/running.html

-------- TO DO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
- Use "net = Network(collect())"
- Use to switch plasticity on or off S = Synapses(..., '''...
                     plastic : boolean (shared)
                     ...''')
"""
import setuptools
import os, sys, pickle, shutil
from brian2 import *
from numpy import *
from time import localtime

import matplotlib as mpl
import matplotlib.pyplot as plt

prefs.codegen.target = 'numpy'

helper_dir = 'helper_functions'
plotting_funcs_dir = 'plotting_functions'

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))
sys.path.append(os.path.join(parent_dir, plotting_funcs_dir))

# Helper modules
from att_net_obj_mdf import AttractorNetwork
from control_plot_learned_attractor import *

# 1 ========== Results/experiment folders ==========

network_sim_dir = os.path.join(
	parent_dir, 
	'network_results')

if not(os.path.isdir(network_sim_dir)):
	os.mkdir(network_sim_dir)

script_results_path = os.path.join(
	network_sim_dir,
	'attractor_stability')

if not(os.path.isdir(script_results_path)):
	os.mkdir(script_results_path)

idt = localtime()  

simulation_id = str(idt.tm_year) + '{:02}'.format(idt.tm_mon) + \
	'{:02}'.format(idt.tm_mday) + '_'+ '{:02}'.format(idt.tm_hour) + '_' + \
	'{:02}'.format(idt.tm_min) + '_' + '{:02}'.format(idt.tm_sec)

# Starts a new scope for magic functions
# start_scope()

exp_name = '_attractor_stability'

# results destination folder
simulation_results_path = os.path.join(script_results_path, simulation_id + exp_name)

if not(os.path.isdir(simulation_results_path)):
	os.mkdir(simulation_results_path)

# 2 ========== Simulation parameters ==========
print('\n> setting up network')

nets = 1 # number of networks

n = AttractorNetwork() # network class

n.t_run = 1*second # simulation time
n.id = simulation_id

# variables for @network_operation function for stimulus change
n.stimulus_pulse = True
n.stimulus_pulse_clock_dt = 1*second # zero seconds not possible
n.stimulus_pulse_duration = n.stimulus_pulse_clock_dt # Set pulse duration

n.int_meth_neur = 'linear'
n.int_meth_syn = 'euler'
n.stim_type_e = 'flat_to_E_fixed_size' # Stimulus type (square, circle, triangle, cross,...)

n.stim_size_e = 64
n.stim_offset_e = 80
n.stim_freq_e = 3900*Hz # 3900*Hz

n.stim_type_i = 'flat_to_I'
n.stim_size_i = n.N_i
n.stim_freq_i = 5000*Hz # 5000*Hz

# Spikemonitors
n.Input_to_E_mon_record = True
n.Input_to_I_mon_record = True
n.E_mon_record = True
n.I_mon_record = True

n.stim_freq_original = n.stim_freq_e

# Learning rule 
n.plasticity_rule = 'LR2' # LR1, LR2
n.parameter = '2.4' #
n.neuron_type = 'LIF' #
n.bistability = True

# Connection weights
n.w_input_e = 1*mV # 1
n.w_input_i = 1*mV # 1
n.w_e_e = 0.5*mV # 0.5
n.w_e_i = 1*mV # 1
n.w_i_e = 1*mV # 1
n.w_input_s = 67*mV # 1

# Other connection weight variables
n.w_e_e_max = 7.5*mV 			# max weight in EE connections

n.fixed_attractor = False # [?]
n.fixed_attractor_wmax = 'all_max' # [?]

n.init_network_modules() # creates/initializes all objects/parameters/mons

n.net.store(
	name = 'initialized_net_' + simulation_id, 
	filename = os.path.join(
		simulation_results_path,
		'initialized_net_' + simulation_id
		)
)

# 3 ========== learning cue ==========

# Name of simulation
n.exp_type = exp_name + '_learning_cue'

# Running simulation
print('\n> forming attractor for cue stimulus\n')

n.run_network(period = 2)

n.net.store(
	name = 'trained_net_' + simulation_id, 
	filename = os.path.join(
		simulation_results_path,
		'trained_net_' + simulation_id
		)
)

# 3.1 - saving and plotting results ==========

fn = os.path.join(
	simulation_results_path,
	n.id +'_' + n.exp_type + '_trained.pickle')

with open(fn, 'wb') as f:
	pickle.dump((
		n.plasticity_rule,
		nets,
		n.path_sim, 
		n.id,
		n.t_run, 
		n.exp_type,
		n.stim_type_e,
		n.stim_size_e,
		n.stim_freq_e,
		n.stim_offset_e,
		n.stim_type_i,
		n.stim_size_i,
		n.stim_freq_i,
		n.stimulus_pulse_duration,
		n.N_input_e, 
		n.N_input_i, 
		n.N_e, 
		n.N_i,
		len(n.stim_inds_original_E),
		n.w_e_e_max,
		n.Input_to_E_mon.t[:],
		n.Input_to_E_mon.i[:],
		n.Input_to_I_mon.t[:],
		n.Input_to_I_mon.i[:],
		n.E_mon.t[:],
		n.E_mon.i[:],
		n.I_mon.t[:],
		n.I_mon.i[:]), f)

print('\n -> pickled to: ', fn)

control_plot_learned_attractor(
	network_state_path = simulation_results_path,
	pickled_data = n.id +'_' + n.exp_type,
	name = '_trained.pickle')

# Neuron activity rasterplot
# plt.plot(n.SE_mon.t/ms, n.SE_mon.i, '.k')
# plt.title('E Spontaneous activity')
# plt.xlabel('Time (ms)')
# plt.ylabel('Neuron index')
# plt.show()

# spk_trains = n.E_mon.spike_trains()

# print('\nPrinting spk trains: \n')

# for x in range(len(spk_trains)):
# 	print('id: ', x, ' | # spks: ', len(spk_trains[x]))

print('\nlearn_stimulus.py - END.\n')


