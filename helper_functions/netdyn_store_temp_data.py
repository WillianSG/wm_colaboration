# -*- coding: utf-8 -*-
"""
@author: slehfeldt

Input:
- var_outer: outer running variable of loop
- var_inner: inner running variable of loop
- cwd: string describing the current working directory
- folder: path to simulation folder
- paths_temp: list of temporary dirs returned by netdyn_create_temp_folders()
- data_temp: list of lists with simulation data (spike times and neuron indices)

Output: 

Comments:
- Stores temporary data of the current loop iteration as single .csv into the belonging temporary simulation folders.
"""
import os
import numpy as np

def netdyn_store_temp_data(var_outer, var_inner, folder, paths_temp, 
	data_temp):    
	# Loop through the temporary simulation folders
	for i in np.arange(0, len(data_temp)):
		data_name = os.path.split(paths_temp[i])[1]

		# Store each list as an individual .csv file into its belonging folder
		with open(os.path.join(paths_temp[i], data_name +'_var_outer_' + format(var_outer, '09.3f') + '_var_inner_' + format(var_inner, '09.3f') + '.csv'), 'wb') as f:
			np.savetxt(f, data_temp[i][0:len(data_temp[i])], fmt = '%.4e',delimiter = ',')
