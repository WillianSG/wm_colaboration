# -*- coding: utf-8 -*-
"""
@author: asonntag

Input:
- N: list with population sizes
- var_range: list with variable range of experiment
- s_tpoints: list with dictionaries of spike times of each population
- n_inds: list with dictionaries of neuron indices of each population
- stim_size: stimulus size / number of neurons forming a stimulus

Output:
- collect_mean_r_{pop} = dictionary with neuron_resolved mean firing rate of a population
- collect_std_r_{pop} = dictionyary with standard deviations of firing 
rates of a population 

Comments:
- Returns dictionaries of mean firing rates (derived from neuron_resolved histograms) and standard deviations of firing rates.
"""
import sys, os
from brian2 import mean, std

helper_dir = 'helper_functions'

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))

# User define modules
from firing_rate_histograms import *

def netdyn_return_means_and_stds_as_dicts(N, var_range, s_tpoints, n_inds,stim_size):
	collect_mean_r_input_e = {}     
	collect_std_r_input_e = {} 
	collect_mean_r_input_i = {}     
	collect_std_r_input_i = {} 
	collect_mean_r_e = {}     
	collect_std_r_e = {}
	collect_mean_r_i = {}     
	collect_std_r_i = {} 

	for var_outer, freq in enumerate(var_range[0]):
		for var_inner, weight in enumerate(var_range[1]):
			# Calculate firing rate histograms for plotting

			# 'neuron_resolved' histograms    
			[input_e_n_hist_count, 
			input_e_n_hist_edgs,
			input_e_n_hist_fr] = firing_rate_histograms(
				tpoints = s_tpoints[0][var_outer, var_inner], 
				inds = n_inds[0][var_outer, var_inner],
				bin_width = None,
				N_pop = stim_size, 
				flag_hist = 'neuron_resolved')

			[input_i_n_hist_count, 
			input_i_n_hist_edgs,
			input_i_n_hist_fr] = firing_rate_histograms(
				tpoints = s_tpoints[1][var_outer, var_inner], 
				inds = n_inds[1][var_outer, var_inner],
				bin_width = None,
				N_pop = N[1], 
				flag_hist = 'neuron_resolved')

			[e_n_hist_count, 
			e_n_hist_edgs, 
			e_n_hist_fr] = firing_rate_histograms(
				tpoints = s_tpoints[2][var_outer, var_inner], 
				inds = n_inds[2][var_outer, var_inner],
				bin_width = None, 
				N_pop = stim_size, 
				flag_hist = 'neuron_resolved')

			[i_n_hist_count, 
			i_n_hist_edgs,
			i_n_hist_fr] = firing_rate_histograms(
				tpoints = s_tpoints[3][var_outer, var_inner], 
				inds = n_inds[3][var_outer, var_inner],
				bin_width = None, 
				N_pop = N[3], 
				flag_hist = 'neuron_resolved')

			# Calculate mean values and standard deviations of histograms

			# 'neuron_resolved' histograms
			n_hist_rate_mean_input_e = mean(input_e_n_hist_fr) 
			n_hist_rate_std_input_e = std(input_e_n_hist_fr)
			n_hist_rate_mean_input_i = mean(input_i_n_hist_fr) 
			n_hist_rate_std_input_i = std(input_i_n_hist_fr)
			n_hist_rate_mean_e = mean(e_n_hist_fr) 
			n_hist_rate_std_e = std(e_n_hist_fr)
			n_hist_rate_mean_i = mean(i_n_hist_fr) 
			n_hist_rate_std_i = std(i_n_hist_fr)

			#collect mean and stds in dictionaries
			collect_mean_r_input_e[var_outer, var_inner] = n_hist_rate_mean_input_e   
			collect_std_r_input_e[var_outer, var_inner] = n_hist_rate_std_input_e

			collect_mean_r_input_i[var_outer, var_inner] = n_hist_rate_mean_input_i
			collect_std_r_input_i[var_outer, var_inner] = n_hist_rate_std_input_i

			collect_mean_r_e[var_outer, var_inner] = n_hist_rate_mean_e
			collect_std_r_e[var_outer, var_inner] = n_hist_rate_std_e
			collect_mean_r_i[var_outer, var_inner] = n_hist_rate_mean_i
			collect_std_r_i[var_outer, var_inner] = n_hist_rate_std_i

	return collect_mean_r_input_e, collect_std_r_input_e, \
		collect_mean_r_input_i, collect_std_r_input_i, collect_mean_r_e, \
		collect_std_r_e, collect_mean_r_i, collect_std_r_i

