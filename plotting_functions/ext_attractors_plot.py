# -*- coding: utf-8 -*-
"""
@author: wgirao
@based-on: asonntag (adapted from Lehfeldt)

Inputs:

Outputs:

Comments:
- Plot simulation data with different stimulus durations
- [WRONG PARAMETER] run and fix
"""

from brian2 import ms, mV, second
import os, sys, pickle
import numpy as np

helper_dir = 'helper_functions'

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))

from learning_plot_spiketrains_and_histograms import *
from learning_plot_rho_matrix_snapshots import *
from learning_plot_w_matrix_snapshots import *
from learning_plot_xpre_matrix_snapshots import *
from learning_plot_xpost_matrix_snapshots import *
from learning_check_attractor_frequency import *
from learning_plot_performance_analysis import *
from learning_check_attractor_wmatrix import *
from learning_check_for_delay_activity import *

# from IPython import get_ipython

def ext_attractors_plot(
	path_sim_folder_superior, 
	sim_id,exp_type, 
	spiketrains_and_histograms,
	rho_matrix_snapshots, 
	w_matrix_snapshots, 
	xpre_matrix_snapshots,
	xpost_matrix_snapshots, 
	learning_performance, 
	check_wmatrices):
	
	# Set path and select simulation folder
	sim_path = 'Simulations/' # [WRONG PARAMETER]
	sim_folder = path_sim_folder_superior

	# Choose types of plots
	if check_wmatrices:
		attractor_wmatrices = {}

	if learning_performance:
		delay_activities = {}
		attractor_frequencies = {}
		simulation_flags_pulse_duration = []

	# Loading data

	cwd = os.getcwd()
	sim_folders_list = sorted(os.listdir(path_sim_folder_superior))

	for i in np.arange(0, len(sim_folders_list[0:-1]), 1):
		sim_data = os.path.join(path_sim_folder_superior, sim_folders_list[i], sim_folders_list[i] + '.pickle')

		with open(sim_data,'rb') as f:(
			path_sim, 
			sim_id, 
			num_networks,
			simulation_flag_pulse_duration,
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
			pulse_durations,
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

		path_snaps = os.path.join(path_sim_folder_superior, 
			sim_id + '_' + exp_type, 
			folder_snaps)

		print('\nPlotting simulation data...\n')

		# ip = get_ipython()
		# ip.run_line_magic('matplotlib', 'inline') # [?]

		# Spiketrains and histograms
		if spiketrains_and_histograms:
			print(' > Plotting spiketrains and histograms...')

			learning_plot_spiketrains_and_histograms(
				sim_id = sim_id,
				path_sim = os.path.join(
					path_sim_folder_superior, 
					sim_folders_list[i]),
				stim_size = len_stim_inds_original_E[0],
				N = [N_input_e, N_input_i, N_e, N_i],
				s_tpoints = [s_tpoints_input_e, s_tpoints_input_i, s_tpoints_e, s_tpoints_i],
				n_inds = [n_inds_input_e, n_inds_input_i, n_inds_e, n_inds_i],
				bin_width_desired = 50*ms,
				t_run = t_run,
				exp_type = exp_type)

		# Rho matrix snapshot
		if rho_matrix_snapshots:
			print(' > Plotting rho matrix snapshots...') 
			learning_plot_rho_matrix_snapshots(
				sim_id = sim_id, 
				path_sim =  os.path.join(
					path_sim_folder_superior, sim_folders_list[i]), 
				path_rho = os.path.join(path_snaps, sim_id + '_rho_snaps'),
				rho_matrix_snapshots_step = rho_matrix_snapshots_step, 
				t_run = t_run, 
				rhomin = 0,
				rhomax = 1, 
				N_e = N_e,
				N_i = N_i)

		# Weight matrix snapshot
		if w_matrix_snapshots:
			print(' > Plotting weight matrix snapshots...')  
			learning_plot_w_matrix_snapshots(
				cwd = cwd,
				sim_id = sim_id,
				path_sim =  os.path.join(
					path_sim_folder_superior, 
					sim_folders_list[i]),
				path_w =  os.path.join(
					path_snaps, 
					sim_id + '_w_matrix_snaps'),
				w_matrix_snapshots_step = w_matrix_snapshots_step,
				t_run = t_run,
				wmin = 0,
				wmax = w_e_e_max/mV,
				N_e = N_e,
				N_i = N_i)

		# Xpre matrix snapshots
		if xpre_matrix_snapshots:
			print(' > Plotting xpre matrix snapshots...')
			learning_plot_xpre_matrix_snapshots(
				cwd = cwd, 
				sim_id = sim_id, 
				path_sim =  os.path.join(path_sim_folder_superior, 
					sim_folders_list[i]), 
				path_xpre =  os.path.join(path_snaps, sim_id + '_xpre_snaps'),
				xpre_matrix_snapshots_step = xpre_matrix_snapshots_step, 
				t_run = t_run, 
				xpremin = 0,
				xpremax = 8, 
				N_e = N_e,
				N_i = N_i)

		# Xpost matrix snapshots
		if xpost_matrix_snapshots:
			print(' > Plotting xpost matrix snapshots...')
			learning_plot_xpost_matrix_snapshots(
				cwd = cwd,
				sim_id = sim_id,
				path_sim =  os.path.join(
					path_sim_folder_superior,
					sim_folders_list[i]),
				path_xpost =  os.path.join(
					path_snaps,
					sim_id + '_xpost_snaps'),
				xpost_matrix_snapshots_step = xpost_matrix_snapshots_step,
				t_run = t_run,
				xpostmin = 0,
				xpostmax = 15,
				N_e = N_e,
				N_i = N_i)

		# Evaluation of learning performance

		if learning_performance:  
			print( '\n Analysing learning performance...')

			delay_activities_temp = learning_check_for_delay_activity(
				s_tpoints_input_e = s_tpoints_input_e,
				n_inds_input_e = n_inds_input_e,
				s_tpoints_e = s_tpoints_e,
				n_inds_e = n_inds_e, 
				t_run = t_run,
				stim_pulse_duration = stim_pulse_duration,
				size_attractor = len_stim_inds_original_E[0],
				plot_spiketrains = False,
				sim_id = sim_id,
				path_sim = os.path.join(
					path_sim_folder_superior, 
					sim_folders_list[i]))

			delay_activities[sim_folders_list[i]] = delay_activities_temp

			attractor_frequency_temp = learning_check_attractor_frequency(
				s_tpoints = s_tpoints_e,
				n_inds = n_inds_e, 
				t_run = t_run,
				stim_pulse_duration = stim_pulse_duration,
				size_attractor = len_stim_inds_original_E[0])

			attractor_frequencies[sim_folders_list[i]] = attractor_frequency_temp

			simulation_flags_pulse_duration.append(simulation_flag_pulse_duration)

		path_snaps = os.path.join(
			path_sim_folder_superior, 
			sim_id + '_' + exp_type, 
			folder_snaps)

		if check_wmatrices:
			attractor_mean_wmat_temp = learning_check_attractor_wmatrix(
				cwd = cwd, 
				sim_id = sim_id, 
				path_sim = os.path.join(
					path_sim_folder_superior, 
					sim_folders_list[i]),
				path_w = os.path.join(path_snaps, sim_id + '_w_matrix_snaps'),
				w_matrix_snapshots_step = w_matrix_snapshots_step,
				t_run = t_run,
				stim_pulse_duration = stim_pulse_duration)

			attractor_wmatrices[sim_folders_list[i]] = attractor_mean_wmat_temp

	# [?]
	if learning_performance:
		[counts, 
		mean_attractor_frequencies,
		std_attractor_frequencies,
		attractor_frequencies] = learning_plot_performance_analysis(
			path_sim = path,
			sim_id = sim_id,
			exp_type = exp_type,
			num_networks = num_networks,
			sim_folders_list = sim_folders_list,
			pulse_durations = pulse_durations/second,
			simulation_flags_pulse_duration = simulation_flags_pulse_duration,
			delay_activities = delay_activities,
			attractor_frequencies = attractor_frequencies)

		# Store learning performance data 
		os.chdir(path_sim_folder_superior)

		fn = os.path.join(
			path_sim_folder_superior, 
			sim_id + '_' + '_learning_performance_data.pickle')

		with open(fn, 'wb') as f:  
			pickle.dump((
			delay_activities,
			attractor_frequencies,
			simulation_flags_pulse_duration,
			counts, 
			mean_attractor_frequencies,
			std_attractor_frequencies,
			attractor_frequencies), f)

	print('\n\n > finished plotting')
















