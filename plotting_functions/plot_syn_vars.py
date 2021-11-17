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

import os, pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_syn_vars(path):
	path_u = 'stimulus_neur_u.pickle'

	pickled_file = os.path.join(path, path_u)

	with open(pickled_file,'rb') as f:(
		us,
		mon_t) = pickle.load(f)

	path_x = 'stimulus_neur_x.pickle'

	pickled_file = os.path.join(path, path_x)

	with open(pickled_file,'rb') as f:(
		xs,
		mon_t) = pickle.load(f)

	plt.plot(mon_t, us[0])
	plt.plot(mon_t, us[1])
	plt.plot(mon_t, us[2])
	plt.plot(mon_t, us[3])
	plt.plot(mon_t, us[4])
	plt.plot(mon_t, us[5])

	plt.plot(mon_t, xs[0])
	plt.plot(mon_t, xs[1])
	plt.plot(mon_t, xs[2])
	plt.plot(mon_t, xs[3])
	plt.plot(mon_t, xs[4])
	plt.plot(mon_t, xs[5])

	plt.savefig(
		os.path.join(path, 'synaptic_variables.png'), 
		bbox_inches = 'tight')

	plt.close()