# -*- coding: utf-8 -*-
"""
@author: slehfeldt

Input:
- N: list with population sizes
- var_range: list with variable range of experiment
- s_tpoints: list with dictionaries of spike times of each population
- n_inds: list with dictionaries of neuron indices of each population

Output:
- isi_input, isi_e, isi_i = dictionaries with lists of inter spike intervals of every loop iteration 

Comments:
- Gets the inter spike intervals of the populations for every loop iteration of the netdyn experiments.
"""
import sys

helper_dir = 'helper_functions'

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))

# User define modules
from inter_spike_intervals import *

def netdyn_get_inter_spike_intervals_as_dicts(N, var_range, s_tpoints, n_inds):
	isi_input_e = {}
	isi_input_i = {}
	isi_e = {}
	isi_i = {}    

	for var_outer, freq in enumerate(var_range[0]):
		for var_inner, weight in enumerate(var_range[1]):
			# ISI Input_to_E
			isi_input_e_list = inter_spike_intervals(N[0], 
				n_inds[0][var_outer, var_inner], 
				s_tpoints[0][var_outer, 
				var_inner]) # Get list of inter spike intervals

			# Store list in dictionary 
			isi_input_e[var_outer, var_inner] = isi_input_e_list

			# ISI Input_to_I
			isi_input_i_list = inter_spike_intervals(N[1], 
			n_inds[1][var_outer, 
			var_inner], 
			s_tpoints[1][var_outer, 
			var_inner]) # Get list of inter spike intervals

			# Store list in dictionary
			isi_input_i[var_outer, var_inner] = isi_input_i_list

			# ISI E
			isi_e_list = inter_spike_intervals(N[2], 
				n_inds[2][var_outer, var_inner], 
				s_tpoints[2][var_outer, var_inner])

			isi_e[var_outer, var_inner] = isi_e_list

			# ISI I
			isi_i_list = inter_spike_intervals(N[3], 
				n_inds[3][var_outer, var_inner], 
				s_tpoints[3][var_outer, var_inner])

			isi_i[var_outer, var_inner] = isi_i_list

	return isi_input_e, isi_input_i, isi_e, isi_i
