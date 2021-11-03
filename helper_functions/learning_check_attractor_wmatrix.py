# -*- coding: utf-8 -*-
"""
@author: slehfeldt

Inputs:
- cwd: string holding the path of the current working directory
- sim_id: simulation identifier
- path_sim: string holding the path to the simulation folder
- path_w: string holding the folder name where snapshots are stored as .csv
- w_matrix_snapshots_step: temporal step of snapshot clock
- t_run: duration of simulation
- wmin/wmax: lower/upper weight boundary
- N_e / N_i: number of neruons in E and I

Outputs:
- mean_wmat: calculated mean of weights belonging to attractor

Comments:
- Extracts the weight matrix snapshot at stimulus offset and calculates the mean weight of the synapses belonging to the attractor.
"""

def learning_check_attractor_wmatrix(cwd, sim_id, path_sim, path_w, w_matrix_snapshots_step, t_run, stim_pulse_duration):
	import os
	import numpy as np
	from brian2 import second, mean

	# Go to directory of w_matrix snapshots and get list of items

	files_list_orig = sorted(os.listdir(path_w))
	files_list = []  

	# get list of .csv files
	for f in files_list_orig:
		if f.endswith('.csv'):
			files_list.append(f)

	# Generate list of timepoints of snapshots
	t = np.arange(0, t_run, w_matrix_snapshots_step)*second

	# detect which file belongs to stimulus_pulse duration  
	file_ind = np.where(t == stim_pulse_duration)  

	# load snapshot at stimulus offset
	wmat = np.loadtxt(os.path.join(path_w,files_list[file_ind[0][0]]), delimiter = ',')

	# take mean
	mean_wmat = 0 if np.isnan(np.mean(wmat)) else np.mean(wmat) 

	return mean_wmat