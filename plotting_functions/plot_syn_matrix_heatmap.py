# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System

Function:
-

Script arguments:
-

Script output:
-
"""

import os, sys, pickle
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

def plot_syn_matrix_heatmap(path_to_data):
	pickled_list = os.listdir(path_to_data)

	for pickled in pickled_list:
		pickled_file = os.path.join(path_to_data, pickled)

		timestep = pickled.split('_')[0]

		with open(pickled_file,'rb') as f:(
			synaptic_matrix) = pickle.load(f)

		plot = plt.imshow(
			synaptic_matrix , 
			cmap = 'bwr' , 
			interpolation = 'nearest')

		plt.title('E_E syn matrix at ' + timestep + ' ms')

		plt.colorbar(plot)

		plt.clim(-1.0, 1.0)

		plt.savefig(
			os.path.join(path_to_data, pickled.replace('.pickle', '.png')), 
			bbox_inches = 'tight')

		plt.close()


