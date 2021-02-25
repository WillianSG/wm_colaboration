# -*- coding: utf-8 -*-
"""
@author: wgirao adapted from asonntag
@original: Lehfeldt

Input:

Output:

Comments:
- Load and plot stored pickle data from learning experiments.
- [?] - y leaving the last file out?
- [wrong] - not array
"""
from brian2 import ms, mV, second
import os, sys, pickle
import numpy as np

helper_dir = 'helper_functions'
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))

# User defined imports
from varying_params_plot_performance_analysis import *
from learning_check_for_delay_activity import *

def varying_params_attractor_analysis_plot(path_sim_folder_superior, sim_id, exp_type,spiketrains_and_histograms = False, w_matrix_snapshots = False):

	# 1 - Choose types of plots
	attractor_wmatrices = {}

	delay_activities = {} # simulation_name, classification result
	simulation_flags_params = []

	# 2 - Loading data

	cwd = os.getcwd()
	sim_folders_list = sorted(os.listdir(path_sim_folder_superior))

	counter = 1

	# [?] - y leaving the last file out?
	for i in np.arange(0, len(sim_folders_list[0:-1]), 1):
		sim_data = os.path.join(path_sim_folder_superior, sim_folders_list[i], sim_folders_list[i] + '.pickle')

		with open(sim_data,'rb') as f:(
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
			stim_pulse_duration,
			varying_params,
			N_input_e, 
			N_input_i, 
			N_e, 
			N_i,
			len_stim_inds_original_E,
			w_e_e_max, 
			path_w, 
			path_xpre,
			path_xpost,
			s_tpoints_input_e,
			s_tpoints_e,
			n_inds_input_e,
			n_inds_e,
			simulation_flag   
			) = pickle.load(f)

		# path_snaps = os.path.join(path_sim_folder_superior, sim_id  + '_' +exp_type, folder_snaps)

		# 3 - Plotting

		# Evaluation of learning performance
		print(' > performing performance analysis')

		delay_activities_temp = learning_check_for_delay_activity(
			s_tpoints_input_e = s_tpoints_input_e,
			n_inds_input_e = n_inds_input_e,
			s_tpoints_e = s_tpoints_e,
			n_inds_e = n_inds_e, 
			t_run = t_run,
			stim_pulse_duration = stim_pulse_duration,
			size_attractor = len_stim_inds_original_E, # [wrong] - not array
			plot_spiketrains = False,
			sim_id = sim_id,
			path_sim = os.path.join(path_sim_folder_superior, sim_folders_list[i]))

		delay_activities[sim_folders_list[i]] = delay_activities_temp

		simulation_flags_params.append(simulation_flag)

	counts = varying_params_plot_performance_analysis(
			path_sim = path_sim_folder_superior,
			sim_id = sim_id,
			exp_type = exp_type,
			num_networks = num_networks,
			sim_folders_list = sim_folders_list,
			simulation_flags = simulation_flags_params,
			varying_params = varying_params,
			delay_activities = delay_activities)

	# Storing data

	file = os.path.join(path_sim_folder_superior, sim_folders_list[i], sim_id + '_' + exp_type + '_learning_performance_data.pickle')

	with open(file, 'wb') as f:  
		pickle.dump((
			delay_activities,
			simulation_flags_params,
			counts), f)

	counter += 1









