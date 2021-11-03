# -*- coding: utf-8 -*-
"""
@author: Lehfeldt with adaptations by wgirao

Input:

Output: 

Comments:
- Load and plot stored pickle data from netdyn experiments.
- neuron_resolved firing rates ?
"""
from brian2 import *
import os, sys, shutil, pickle
from time import localtime

helper_dir = 'helper_functions'
plotting_funcs_dir = 'plotting_functions'
net_sim_dir = 'net_simulation'

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))
sys.path.append(os.path.join(parent_dir, plotting_funcs_dir))

from netdyn_plot_spiketrains_and_histograms import *
from netdyn_plot_colourgrid_mean_rates_and_stds import *
from netdyn_plot_input_output_curve import *
from netdyn_plot_inter_spike_interval_histograms import *

from netdyn_return_means_and_stds_as_dicts import *
from netdyn_get_inter_spike_intervals_as_dicts import *

def netdyn_weights_makeplots(path_sim, sim_id, exp_type):
	# Set path, select simulation folder and data, define figure name for result plot

	sim_path = os.path.join(parent_dir, net_sim_dir)
	sim_data =  os.path.join(path_sim, sim_id + '_' + exp_type + '.pickle')

	# Load data from simulation
	with open(sim_data,'rb') as f:
		(path_sim, 
		sim_id, 
		t_run, 
		exp_type,
		simulation_flag,
		stim_type_e,
		stim_size_e,
		stim_freq_e,
		stim_offset_e,
		stim_deg_type_e,
		stim_type_i,
		stim_size_i,
		stim_freq_i,
		N_input_e, 
		N_input_i, 
		N_e, 
		N_i,
		w1_range,
		w1_step, 
		w2_range, 
		w2_step, 
		s_tpoints_input_e, 
		s_tpoints_input_i, 
		s_tpoints_e, 
		s_tpoints_i,
		n_inds_input_e, 
		n_inds_input_i, 
		n_inds_e, 
		n_inds_i) = pickle.load(f)

	# Plotting
	
	mean_rates_and_stds = True
	spiketrains_and_histograms = False
	input_output_curve = True
	inter_spike_intervals = False

    # Spike-trains and firing rates histograms
	if spiketrains_and_histograms:
		netdyn_plot_spiketrains_and_histograms(
			full_path = path_sim, 
			sim_id = sim_id,
			N = [N_input_e, N_input_i, N_e, N_i], 
			stim_size = stim_size_e,
			var_range = [w1_range, w2_range], 
			s_tpoints = [s_tpoints_input_e, s_tpoints_input_i, s_tpoints_e, s_tpoints_i],
			n_inds = [n_inds_input_e, n_inds_input_i, n_inds_e, n_inds_i],
			t_run = t_run,
			bin_width_desired = 0.01, 
			exp_type = exp_type,
			flag_savefig = 'weight_weight')

	# Return neuron_resolved firing rates as dictionary
	if mean_rates_and_stds or input_output_curve:
		# Get mean values and standard deviations
		[rates_mean_input_e, 
		rates_std_input_e, 
		rates_mean_input_i, 
		rates_std_input_i, 
		rates_mean_e, 
		rates_std_e, 
		rates_mean_i, 
		rates_std_i] = netdyn_return_means_and_stds_as_dicts(
			N = [N_input_e, N_input_i, N_e, N_i], 
			var_range = [w1_range, w2_range], 
			s_tpoints = [s_tpoints_input_e, s_tpoints_input_i, s_tpoints_e, s_tpoints_i],
			n_inds = [n_inds_input_e, n_inds_input_i, n_inds_e, n_inds_i],
			stim_size = stim_size_e)

	# Mean rates and standard deviations as colour grid
	if mean_rates_and_stds:
		# Plot colour grid of mean firing rates and standard deviations
		netdyn_plot_colourgrid_mean_rates_and_stds(
			full_path = path_sim, 
			sim_id = sim_id,
			var_range = [w1_range/mV, w2_range/mV], 
			var_steps = [w1_step/mV, w2_step/mV],
			dicts_means = [rates_mean_input_e, rates_mean_input_i, rates_mean_e, rates_mean_i],
			dicts_stds = [rates_std_input_e, rates_std_input_i, rates_std_e, rates_std_i],
			exp_type = exp_type, xlabel = '$w_{EI}$ (mV)', 
			ylabel = ['$w_{IE}$ (mV)', '$w_{IE}$ (mV)'])

	# Frequency input output curve
	if input_output_curve:
		# Plot input-output curve                                             
		netdyn_plot_input_output_curve(
			full_path = path_sim, 
			sim_id = sim_id, 
			N = [N_input_e, N_input_i, N_e, N_i],
			var_range = [w1_range, w2_range], 
			var_steps = [w1_step, w2_step],
			dicts_means = [rates_mean_input_e, rates_mean_input_i, rates_mean_e, rates_mean_i],
			dicts_stds = [rates_std_input_e, rates_std_input_i, rates_std_e, rates_std_i],
			exp_type = exp_type,
			simulation_flag = simulation_flag)

	# Frequency input output curve
	if inter_spike_intervals:
		# Get inter spike intervals from spike trains    
		[ISI_input_e,
		ISI_input_i,
		ISI_e, 
		ISI_i] = netdyn_get_inter_spike_intervals_as_dicts(
			N = [N_input_e, N_input_i, N_e, N_i], 
			var_range = [w1_range, w2_range], 
			s_tpoints = [s_tpoints_input_e, s_tpoints_input_i, s_tpoints_e, s_tpoints_i],
			n_inds = [n_inds_input_e, n_inds_input_i, n_inds_e, n_inds_i])

		# Plot inter spike interval histograms
		ISI = [ISI_input_e, ISI_input_i, ISI_e, ISI_i]
		
		netdyn_plot_inter_spike_interval_histograms(
			full_path = path_sim, 
			sim_id = sim_id,
			var_range = [w1_range, w2_range], 
			isi = ISI, 
			exp_type = exp_type, 
			flag_isi = 'count', 
			flag_savefig = 'freq_weight')





