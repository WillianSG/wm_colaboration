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
import matplotlib.pyplot as plt

simulation_data = '20210307_23_23_34_attractor_stability'

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
simulation_data_path = os.path.join(
	parent_dir, 
	'network_results', 
	'attractor_stability',
	simulation_data,
	'synaptic_matrix_snaps')

pickled_list = sorted(os.listdir(simulation_data_path))

for x in range(0, len(pickled_list)):
	title = int(pickled_list[x].split('_')[-1].split('.')[0].replace('sec', ''))/1000.0
	img_name = pickled_list[x].split('_')[-1].split('.')[0] + '.png'

	sim_data = os.path.join(simulation_data_path, pickled_list[x])

	with open(sim_data,'rb') as f:(
		synaptic_matrix) = pickle.load(f)

	plt.imshow(synaptic_matrix, cmap='coolwarm')
	plt.colorbar()
	plt.title(str(title) + ' sec')
	plt.savefig(os.path.join(simulation_data_path, img_name))
	plt.close()

print('\ncontrol_plot_synaptic_matrix_heatmap.py - END.\n')

