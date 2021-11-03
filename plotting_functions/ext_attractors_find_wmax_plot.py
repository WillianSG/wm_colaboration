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
from learning_plot_spiketrains_and_histograms import *
from learning_plot_rho_matrix_snapshots import *
from learning_plot_w_matrix_snapshots import *
from learning_plot_xpre_matrix_snapshots import *
from learning_plot_xpost_matrix_snapshots import *

from find_wmax_for_attractors_plot_performance_analysis import *

from learning_check_attractor_frequency import *
from learning_check_attractor_wmatrix import *
from learning_check_for_delay_activity import *

def ext_attractors_find_wmax_plot(path_sim_folder_superior, sim_id, exp_type,spiketrains_and_histograms, w_matrix_snapshots):

	# 1 - Choose types of plots

	learning_performance = True
	check_wmatrices = False
	attractor_wmatrices = {}

	if learning_performance:
		delay_activities = {} # simulation_name, classification result
		attractor_frequencies = {} # simulation_name, frequency
		simulation_flags_wmax = []  # [?]

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
			simulation_flag_wmax,
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
			wmax_range,
			N_input_e, 
			N_input_i, 
			N_e, 
			N_i,
			len_stim_inds_original_E,
			w_e_e_max, 
			rho_matrix_snapshots_step,
			folder_snaps,
			w_matrix_snapshots_step, 
			path_w,
			xpre_matrix_snapshots_step, 
			path_xpre,
			xpost_matrix_snapshots_step, 
			path_xpost,
			s_tpoints_input_e, 
			s_tpoints_input_i, 
			s_tpoints_e, 
			s_tpoints_i,
			n_inds_input_e,
			n_inds_input_i,
			n_inds_e, 
			n_inds_i,     
			) = pickle.load(f)

		path_snaps = os.path.join(path_sim_folder_superior, sim_id  + '_' +exp_type, folder_snaps)

		# 3 - Plotting

		print('\nPlotting simulation data...\n')

		# Spiking trains and firing rates histograms
		if spiketrains_and_histograms:
			print(' > plotting spiketrains and histograms')

			learning_plot_spiketrains_and_histograms(
				sim_id = sim_id,
				path_sim = os.path.join(path_sim_folder_superior, sim_folders_list[i]),
				stim_size = len_stim_inds_original_E, # [wrong] - not array
				N = [N_input_e, N_input_i, N_e, N_i],
				s_tpoints = [s_tpoints_input_e, s_tpoints_input_i, s_tpoints_e, s_tpoints_i],
				n_inds = [n_inds_input_e, n_inds_input_i, n_inds_e, n_inds_i],
				bin_width_desired = 50*ms,
				t_run = t_run,
				exp_type = exp_type)

		# Weight matrix snapshots
		if w_matrix_snapshots:
			print(' > plotting w matrix snapshots' )

			learning_plot_w_matrix_snapshots(
				cwd = cwd, 
				sim_id = sim_id, 
				path_sim =  os.path.join(path_sim_folder_superior, sim_folders_list[i]) , 
				path_w =  os.path.join(path_snaps, sim_id+'_w_matrix_snaps') , 
				w_matrix_snapshots_step = w_matrix_snapshots_step, 
				t_run = t_run, 
				wmin = 0,
				wmax = w_e_e_max/mV, 
				N_e = N_e,
				N_i = N_i)

		# Evaluation of learning performance

		if learning_performance:  
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

			attractor_frequency_temp = learning_check_attractor_frequency(
				s_tpoints = s_tpoints_e,
				n_inds = n_inds_e, 
				t_run = t_run,
				stim_pulse_duration = stim_pulse_duration,
				size_attractor = len_stim_inds_original_E) # [wrong] - not array

			attractor_frequencies[sim_folders_list[i]] = attractor_frequency_temp

		if check_wmatrices:
			attractor_mean_wmat_temp = learning_check_attractor_wmatrix(
				cwd = cwd, 
				sim_id = sim_id, 
				path_sim = os.path.join(path_sim_folder_superior, sim_folders_list[i]),
				path_w = os.path.join(path_snaps, sim_id + '_w_matrix_snaps'), 
				w_matrix_snapshots_step = w_matrix_snapshots_step, 
				t_run = t_run,
				stim_pulse_duration = stim_pulse_duration)

			attractor_wmatrices[path_sim_folder_superior] = attractor_mean_wmat_temp

		simulation_flags_wmax.append(simulation_flag_wmax)

	if learning_performance:  
		[counts, mean_attractor_frequencies, std_attractor_frequencies,		attractor_frequencies_classified] = find_wmax_for_attractors_plot_performance_analysis(
				path_sim = path_sim_folder_superior,
				sim_id = sim_id,
				exp_type = exp_type,
				num_networks = num_networks,
				sim_folders_list = sim_folders_list,
				simulation_flags_wmax = simulation_flags_wmax,
				wmax_range = wmax_range/mV,
				delay_activities = delay_activities,
				attractor_frequencies = attractor_frequencies)

	# Storing data

	file = os.path.join(path_sim_folder_superior, sim_folders_list[i], sim_id + '_' + exp_type + '_learning_performance_data.pickle')

	with open(file, 'wb') as f:  
		pickle.dump((
			delay_activities,
			attractor_frequencies,
			simulation_flags_wmax,
			counts, 
			mean_attractor_frequencies,
			std_attractor_frequencies,
			attractor_frequencies), f)

	counter += 1









