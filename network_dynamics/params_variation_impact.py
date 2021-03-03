# -*- coding: utf-8 -*-
"""
@author: wgirao

Input:

Output:

Comments:
"""
import setuptools
import os, sys, pickle, shutil
import warnings
from brian2 import *
from numpy import *
from time import localtime

warnings.filterwarnings('ignore') 

prefs.codegen.target = 'numpy'

helper_dir = 'helper_functions'
plotting_funcs_dir = 'plotting_functions'

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))
sys.path.append(os.path.join(parent_dir, plotting_funcs_dir))

# Helper modules
from att_net_obj import AttractorNetwork
from varying_params_attractor_analysis_plot import *

# 1 - Simulation results folder

exp_name = '_param_variation'
varying_params = ['wmax', 'c', 'tau_pre', 'tau_post', 'rho_neg', 'rho_neg2', 'thr_post', 'thr_pre']

simulation_folder = os.path.join(parent_dir, 'network_results')

if not(os.path.isdir(simulation_folder)):
	os.mkdir(simulation_folder)

idt = localtime()  

simulation_id = str(idt.tm_year) + '{:02}'.format(idt.tm_mon) + \
	'{:02}'.format(idt.tm_mday) + '_'+ '{:02}'.format(idt.tm_hour) + '_' + \
	'{:02}'.format(idt.tm_min) + '_' + '{:02}'.format(idt.tm_sec)

sim_results_folder = os.path.join(simulation_folder, simulation_id + exp_name)

if not(os.path.isdir(sim_results_folder)):
	os.mkdir(sim_results_folder)

# 2 - Simulation settings

num_networks = 10

sim_duration = 8*second			# Duration of each simulation
pulse_duration = 2*second			# Stimulus pulse duration

# 3 - Net initialization
for i in np.arange(0, num_networks, 1):
	print('\n- Simulating network | ', i+1, '/', num_networks)
	    
	n = AttractorNetwork()

	n.t_run = sim_duration 
	n.id = simulation_id

	n.int_meth_neur = 'linear'
	n.int_meth_syn = 'euler' 

	n.stim_type_e = 'flat_to_E_fixed_size'
	n.stim_size_e = 64
	n.stim_freq_e = 3900*Hz 
	n.stim_offset_e = 96

	n.stim_type_i = 'flat_to_I'
	n.stim_size_i = n.N_i
	n.stim_freq_i = 5000*Hz
	  
	# Spikemonitors
	n.Input_to_E_mon_record = True
	n.Input_to_I_mon_record = True
	n.E_mon_record = True
	n.I_mon_record = True

	# @network_operation settings

	# Weight snapshots
	n.w_matrix_snapshots = False

	# variables for @network_operation function for stimulus change
	n.stimulus_pulse = True
	n.stimulus_pulse_clock_dt = pulse_duration

	n.stim_freq_original = n.stim_freq_e

	# Learning rule 
	n.plasticity_rule = 'LR2' # 'LR1', 'LR2'
	n.parameter = '2.4'
	n.neuron_type = 'LIF'
	n.net_size = 'net'
	n.bistability = True

	# Connection weights
	n.w_input_e = 1*mV      
	n.w_input_i = 1*mV
	n.w_e_e = 0.5*mV          
	n.w_e_i = 1*mV          
	n.w_i_e = 1*mV          

	# Other connection weight variables
	n.w_e_e_max = 7.5*mV

	n.fixed_attractor = True              
	n.fixed_attractor_wmax = 'all_max'  

	n.init_network_modules() # creates/initializes all objects/parameters/mons
	  
	# Store initial network state
	n.net.store(name = 'network_' + simulation_id + '_initial_state', filename = os.path.join(sim_results_folder, 'network_' + simulation_id + '_initial_state'))

	# 3.1 - Parameter variation loop

	# restoring network state
	for p in range(0, len(varying_params)):
		n.net.restore(
			name = 'network_' + simulation_id + '_initial_state', 
			filename = os.path.join(sim_results_folder, 'network_' + simulation_id + '_initial_state'))

		n.exp_type = 'learning_net_' + str(i+1) + '_varying_' + str(varying_params[p])

		# 3.1.1 Vary parameter and simulate

		# n.set_varying_param(varying_params[p]) # all params with same new val
		n.vary_param_per_synapse(varying_params[p]) # each synapse with new val

		simulation_flag = [p, varying_params[p]]

		n.run_net()

		# 3.1.2 Storing simulation data

		# a - Timepoints of spikes (s_tpoints) and neuron indices (n_inds)

		# Input_to_E population 
		s_tpoints_input_e = n.Input_to_E_mon.t[:]
		n_inds_input_e = n.Input_to_E_mon.i[:]

		# Input_to_I population 

		# Excitatory population
		s_tpoints_e = n.E_mon.t[:]
		n_inds_e = n.E_mon.i[:]

		# Inhibitory population

		s_tpoints = [s_tpoints_input_e, s_tpoints_e]

		n_inds = [n_inds_input_e, n_inds_e]

		# Pickling (store structure)

		path_sim = n.path_sim
		sim_id = n.id
		t_run = n.t_run
		exp_type = n.exp_type

		stim_type_e = n.stim_type_e
		stim_size_e = n.stim_size_e
		stim_freq_e = n.stim_freq_e
		stim_offset_e = n.stim_offset_e
		stim_type_i = n.stim_type_i
		stim_size_i = n.stim_size_i
		stim_freq_i = n.stim_freq_i
		stim_pulse_duration = n.stimulus_pulse_duration

		N_input_e = n.N_input_e
		N_input_i = n.N_input_i
		N_e = n.N_e
		N_i = n.N_i

		len_stim_inds_original_E = len(n.stim_inds_original_E)
		w_e_e_max = n.w_e_e_max

		path_rho = n.path_snapshots
		path_w = n.path_snapshots

		path_xpre = n.path_snapshots
		path_xpost = n.path_snapshots
		plasticity_rule = n.plasticity_rule

		fn = os.path.join(n.path_sim, n.id + '_' + n.exp_type + '.pickle')

		with open(fn, 'wb') as f:
			pickle.dump((
			path_sim, 
			sim_id,
			num_networks,
			t_run,
			exp_type,
			stim_type_e, 
			stim_type_e,
			stim_size_e,
			stim_freq_e,
			stim_offset_e,
			stim_type_i,
			stim_size_i,
			stim_freq_i,
			stim_pulse_duration,#
			varying_params,
			N_input_e, 
			N_input_i, 
			N_e, 
			N_i,
			len_stim_inds_original_E,#
			w_e_e_max, 
			path_w, 
			path_xpre,
			path_xpost,
			s_tpoints_input_e, #
			s_tpoints_e, #
			n_inds_input_e, #
			n_inds_e,
			simulation_flag), f) #

		print('\nSimulation data pickled to ', fn)

		# Move current simulation folder to superior simulation folder
		shutil.move(n.path_sim, sim_results_folder)

varying_params_attractor_analysis_plot(
	sim_results_folder, 
	sim_id, 
	exp_type)

print('\nparams_variation_impact.py - END.\n')

