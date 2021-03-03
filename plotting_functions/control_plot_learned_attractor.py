# -*- coding: utf-8 -*-
"""
@author: wgirao

Inputs:

Outputs:

Comments:
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

def control_plot_learned_attractor(
	network_state_path = '', 
	pickled_data = '', 
	name = ''):
	sim_data = os.path.join(network_state_path, pickled_data + name) # loading data

	with open(sim_data,'rb') as f:(
		plasticity_rule,
		nets,
		path_sim, 
		sim_id,
		t_run, 
		exp_type,
		stim_type_e,
		stim_size_e,
		stim_freq_e,
		stim_offset_e,
		stim_type_i,
		stim_size_i,
		stim_freq_i,
		stimulus_pulse_duration,
		N_input_e, 
		N_input_i, 
		N_e, 
		N_i,
		len_stim_inds_original_E,
		w_e_e_max,
		s_tpoints_input_e,
		n_inds_input_e,
		s_tpoints_input_i,
		n_inds_input_i,
		s_tpoints_e,
		n_inds_e,
		s_tpoints_i,
		n_inds_i) = pickle.load(f)

	print(' -> plotting spiketrains and histograms')

	learning_plot_spiketrains_and_histograms(
		sim_id = sim_id,
		path_sim = os.path.join(network_state_path, pickled_data),
		stim_size = len_stim_inds_original_E,
		N = [N_input_e, N_input_i, N_e, N_i],
		s_tpoints = [s_tpoints_input_e, s_tpoints_input_i, s_tpoints_e, s_tpoints_i],
		n_inds = [n_inds_input_e, n_inds_input_i, n_inds_e, n_inds_i],
		bin_width_desired = 50*ms,
		t_run = t_run,
		exp_type = exp_type)
















