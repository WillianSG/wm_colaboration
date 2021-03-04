# -*- coding: utf-8 -*-
"""
@author: wgirao adapted from asonntag

Input:

Output:

Comments:
- Investigates the best maximum weight.
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

simulation_folder = os.path.join(parent_dir, 'net_simulation')

sim_results_folder = os.path.join(
	simulation_folder, 
	'20210303_16_05_48_attractor_stability_')

# Helper modules
from att_net_obj import AttractorNetwork
from ext_attractors_find_wmax_plot import *

print('\n> loading data...')
	    
n = AttractorNetwork()

# Restore network state with loaded learning rule parameters       
n.net.restore(
	name = 'network_20210303_16_05_48_state_after_learning', 
	filename = os.path.join(
		sim_results_folder, 
		'network_20210303_16_05_48_state_after_learning')
	)

# 	for w in np.arange(0, len(wmax_range), 1):
# 		n.exp_type = 'learning_network_' + str(i+1) + '_wmax_' + str(wmax_range[w])

# 		# === Set wmax of fixed attractor

# 		n.w_e_e_max = wmax_range[w]
# 		n.set_weights()

# 		simulation_flag_wmax = [w, n.w_e_e_max] # Simulation flags

# 		# === Running net simulation

# 		print('\n > simulating network ', i+1, ' - wmax fixed attractor ', n.w_e_e_max)

# 		n.run_net()

# 		print('[finished] | ', i+1, '- wmax fixed attractor ', n.w_e_e_max)


# 		# === Storing simulation data

# 		# a - Timepoints of spikes (s_tpoints) and neuron indices (n_inds)

# 		# Input_to_E population 
# 		s_tpoints_input_e = n.Input_to_E_mon.t[:]
# 		n_inds_input_e = n.Input_to_E_mon.i[:]

# 		# Input_to_I population 
# 		s_tpoints_input_i = n.Input_to_I_mon.t[:]
# 		n_inds_input_i = n.Input_to_I_mon.i[:]

# 		# Excitatory population
# 		s_tpoints_e = n.E_mon.t[:]
# 		n_inds_e = n.E_mon.i[:]

# 		# Inhibitory population
# 		s_tpoints_i = n.I_mon.t[:]
# 		n_inds_i = n.I_mon.i[:]

# 		s_tpoints = [s_tpoints_input_e, s_tpoints_input_i, s_tpoints_e, s_tpoints_i]

# 		n_inds = [n_inds_input_e, n_inds_input_i, n_inds_e, n_inds_i]

# 		# Pickling (store structure)

# 		path_sim = n.path_sim
# 		sim_id = n.id
# 		t_run = n.t_run
# 		exp_type = n.exp_type

# 		stim_type_e = n.stim_type_e
# 		stim_size_e = n.stim_size_e
# 		stim_freq_e = n.stim_freq_e
# 		stim_offset_e = n.stim_offset_e
# 		stim_type_i = n.stim_type_i
# 		stim_size_i = n.stim_size_i
# 		stim_freq_i = n.stim_freq_i
# 		stim_pulse_duration = n.stimulus_pulse_duration

# 		N_input_e = n.N_input_e
# 		N_input_i = n.N_input_i
# 		N_e = n.N_e
# 		N_i = n.N_i

# 		len_stim_inds_original_E = len(n.stim_inds_original_E)
# 		w_e_e_max = n.w_e_e_max
# 		rho_matrix_snapshots_step = n.rho_matrix_snapshots_step
# 		folder_snaps = os.path.split(n.path_snapshots)[1]

# 		path_rho = n.path_snapshots
# 		w_matrix_snapshots_step = n.w_matrix_snapshots_step
# 		path_w = n.path_snapshots
# 		xpre_matrix_snapshots_step = n.xpre_matrix_snapshots_step

# 		path_xpre = n.path_snapshots
# 		xpost_matrix_snapshots_step = n.xpost_matrix_snapshots_step
# 		path_xpost = n.path_snapshots
# 		plasticity_rule = n.plasticity_rule

# 		fn = os.path.join(n.path_sim, n.id + '_' + n.exp_type + '.pickle')

# 		with open(fn, 'wb') as f:
# 			pickle.dump((
# 			path_sim, 
# 			sim_id, 
# 			num_networks,
# 			simulation_flag_wmax,
# 			t_run, 
# 			exp_type,
# 			stim_type_e, 
# 			stim_type_e,
# 			stim_size_e,
# 			stim_freq_e,
# 			stim_offset_e,
# 			stim_type_i,
# 			stim_size_i,
# 			stim_freq_i,
# 			stim_pulse_duration,
# 			wmax_range,
# 			N_input_e, 
# 			N_input_i, 
# 			N_e, 
# 			N_i,
# 			len_stim_inds_original_E,
# 			w_e_e_max, 
# 			rho_matrix_snapshots_step,
# 			folder_snaps,
# 			w_matrix_snapshots_step, 
# 			path_w,
# 			xpre_matrix_snapshots_step, 
# 			path_xpre,
# 			xpost_matrix_snapshots_step, 
# 			path_xpost,
# 			s_tpoints_input_e, 
# 			s_tpoints_input_i, 
# 			s_tpoints_e, 
# 			s_tpoints_i,
# 			n_inds_input_e,
# 			n_inds_input_i,
# 			n_inds_e, 
# 			n_inds_i), f)

# 		print('\nSimulation data pickled to ', fn)

# 		# Move current simulation folder to superior simulation folder
# 		shutil.move(n.path_sim, sim_results_folder)

# ext_attractors_find_wmax_plot(
# 	sim_results_folder, 
# 	sim_id, 
# 	exp_type, 
# 	spiketrains_and_histograms = True,
# 	w_matrix_snapshots = False)

print('\ntesting - END.\n')

