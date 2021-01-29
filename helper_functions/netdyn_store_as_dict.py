# -*- coding: utf-8 -*-
"""
@author: slehfeldt

Input:
- range_outer: value range of outer running variable of loop
- range_inner: value range of inner running variable of loop
- cwd: string describing the current working directory
- folder: path to simulation folder
- paths_temp: list of temporary folders returned by netdyn_create_temp_folders()

Output: 
- dicts: single dictionaries holding the spike times and neuron indices of each population over the applied value range of running variables. Keywords in dictionaries are indices of loop iterations.

Comments:
- Moves data from single .csv into dictionaries.
"""
import os
import numpy as np

def netdyn_store_as_dict(range_outer, range_inner, cwd, folder, paths_temp, add_Ext_att):
	# Empty dictionaries for storage of simulation data
	# Timepoints of spikes (s_tpoints) and indices of neurons (n_inds) 

	# Input_to_E population    
	dict_s_tpoints_input_e = {}
	dict_n_inds_input_e = {}

	# Input_to_I population    
	dict_s_tpoints_input_i = {}
	dict_n_inds_input_i = {}

	# Excitatory population
	dict_s_tpoints_e = {}
	dict_n_inds_e = {}

	# Inhibitory population
	dict_s_tpoints_i = {}
	dict_n_inds_i = {}

	# Ext_att ecitatory population
	dict_s_tpoints_ext_att = {}
	dict_n_inds_ext_att = {}

	#Wmat dictionaries
	dicts_wmat_E_att_orig = {} 
	dicts_wmat_E_att_cut = {} 
	dicts_wmat_Ext_att = {}

	if add_Ext_att:
		# List of dictionaries for loop
		dicts = [dict_s_tpoints_input_e, dict_n_inds_input_e,
		dict_s_tpoints_input_i, dict_n_inds_input_i,
		dict_s_tpoints_e,dict_n_inds_e,
		dict_s_tpoints_i,dict_n_inds_i,
		dict_s_tpoints_ext_att,dict_n_inds_ext_att,
		dicts_wmat_E_att_orig,dicts_wmat_E_att_cut,dicts_wmat_Ext_att]
	else:
		# List of dictionaries for loop
		dicts = [dict_s_tpoints_input_e, dict_n_inds_input_e,
		dict_s_tpoints_input_i, dict_n_inds_input_i,
		dict_s_tpoints_e,dict_n_inds_e,
		dict_s_tpoints_i,dict_n_inds_i]

	# Counter that is incremented after every r_inner loop is finished: counts number of running variable combinations (r_outer vs r_inner)
	# Controls the correct read-out of .csv-file from list of items in current folder
	counter = 0

	# For every running variable combination: laod stored data into dictionary
	for r_outer in range(0, len(range_outer)):
		for r_inner in range(0, len(range_inner)):
			# Loop through temporary folders
			for p in np.arange(0, len(paths_temp)):
				d = dicts[p]

				# Get list of items in current folder
				files_temp = sorted(os.listdir(paths_temp[p]))

				abs_file_path = os.path.join(paths_temp[p], 
					files_temp[counter])

				# Load data from file to position in dictionary
				d[r_outer,r_inner] = np.loadtxt(abs_file_path, delimiter = ',')
			counter += 1

	if add_Ext_att:
		return dicts[0], dicts[1], dicts[2], dicts[3], dicts[4], dicts[5], dicts[6], dicts[7], dicts[8], dicts[9], dicts[10], dicts[11], dicts[12]
	else:
		return dicts[0], dicts[1], dicts[2], dicts[3], dicts[4], dicts[5], dicts[6], dicts[7]