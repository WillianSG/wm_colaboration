# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
- Learning of stimulus in an attractor network.
- https://brian2.readthedocs.io/en/stable/user/running.html
- Plotting scripts consume the pickled data

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
print('\n> initializing network')

nets = 1 				# number of networks

n = AttractorNetwork() 	# network class

n.t_run = 1*second 		# simulation time
n.id = simulation_id

# variables for @network_operation function for stimulus change
n.stimulus_pulse = True
n.stimulus_pulse_clock_dt = 1*second # zero seconds not possible
n.stimulus_pulse_duration = n.stimulus_pulse_clock_dt # Set pulse duration

n.int_meth_neur = 'linear'
n.int_meth_syn = 'euler'
n.stim_type_e = 'flat_to_E_fixed_size' # square, circle, triangle, cross,..

n.stim_size_e = 64
n.stim_offset_e = 80
n.stim_freq_e = 3900*Hz 	# 3900*Hz

n.stim_type_i = 'flat_to_I'
n.stim_size_i = n.N_i
n.stim_freq_i = 5000*Hz 	# 5000*Hz

# Spikemonitors
n.Input_to_E_mon_record = True
n.Input_to_I_mon_record = True
n.E_mon_record = True
n.I_mon_record = True

n.stim_freq_original = n.stim_freq_e

# Learning rule 
n.plasticity_rule = 'LR2' 	# LR1, LR2
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
n.w_e_e_max = 7.5*mV 		# max weight in EE connections

n.fixed_attractor = False # [?]
n.fixed_attractor_wmax = 'all_max' # [?]

n.init_network_modules() # creates/initializes all objects/parameters/mons

# 2.1 saving net initialized state ==========

n.net.store(
	name = 'initialized_net_' + simulation_id, 
	filename = os.path.join(
		simulation_results_path,
		'initialized_net_' + simulation_id
		)
)

# 3 ========== learning cue ==========

n.exp_type = exp_name + '_learning_cue'

print('\n> forming attractor for cue stimulus\n')

n.save_monitors() 			# monitors for current sim

n.run_network(period = 2) 	# Running simulation

# 3.1 saving net trained on cue state ==========
n.net.store(
	name = 'trained_cue_net_' + simulation_id, 
	filename = os.path.join(
		simulation_results_path,
		'trained_cue_net_' + simulation_id
		)
)

# 3.2 - pickling and plotting ==========

fn = os.path.join(
	simulation_results_path,
	n.id +'_' + n.exp_type + '_trained.pickle')

# 'spk_mons' gets [input_to_E, input_to_I, E_mon, I_mon]
spk_mons = n.get_stimulus_monitors()

# print(spk_mons[0].i[:])

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
		spk_mons[0].t[:],
		spk_mons[0].i[:],
		spk_mons[1].t[:],
		spk_mons[1].i[:],
		spk_mons[2].t[:],
		spk_mons[2].i[:],
		spk_mons[3].t[:],
		spk_mons[3].i[:]), f)

print('\n -> pickled to: ', fn)

control_plot_learned_attractor(
	network_state_path = simulation_results_path,
	pickled_data = n.id +'_' + n.exp_type,
	name = '_trained.pickle')

# 4 ========== learning followup stimuli ==========

n.t_run = 3*second
n.stimulus_pulse_clock_dt = 2*second
n.stimulus_pulse_duration = 2*second

n.change_stimulus_e(stimulus = 'square')

n.exp_type = exp_name + '_learning_' + n.stim_type_e

print('\n> forming attractor for new stimulus [ ', n.stim_type_e, ' ]\n')

n.save_monitors()

n.run_network(period = 2)

# 4.1 saving net trained on cue state ==========
n.net.store(
	name = 'trained_' + n.stim_type_e + '_net_' + simulation_id, 
	filename = os.path.join(
		simulation_results_path,
		'trained_' + n.stim_type_e + '_net_' + simulation_id
		)
)

# 4.2 - pickling and plotting ==========

fn = os.path.join(
	simulation_results_path,
	n.id +'_' + n.exp_type + '_trained.pickle')

spk_mons = n.get_stimulus_monitors()

# print(spk_mons[0].i[:])

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
		spk_mons[0].t[:],
		spk_mons[0].i[:],
		spk_mons[1].t[:],
		spk_mons[1].i[:],
		spk_mons[2].t[:],
		spk_mons[2].i[:],
		spk_mons[3].t[:],
		spk_mons[3].i[:]), f)

print('\n -> pickled to: ', fn)

control_plot_learned_attractor(
	network_state_path = simulation_results_path,
	pickled_data = n.id +'_' + n.exp_type,
	name = '_trained.pickle')

print('\nlearn_stimulus.py - END.\n')


