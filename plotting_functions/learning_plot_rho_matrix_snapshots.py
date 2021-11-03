# -*- coding: utf-8 -*-
"""
@author: asonntag adapted from Lehfeldt

Inputs:
- sim_id: simulation identifier
- path_sim: string holding the path to the simulation folder
- path_rho: string holding the folder name where snapshots are stored as .csv
- rho_matrix_snapshots_step: temporal step of snapshot clock 
- t_run: duration of simulation
- rhomin/rhomax: lower/upper weight boundary
- N_e / N_i: number of neruons in E and I

Outputs:

Comments:
- Plots snapshots of the rho matrix as a colour grid (x-axis = source neurons, y-axis = target neurons).
"""

def learning_plot_rho_matrix_snapshots(sim_id, path_sim, path_rho, rho_matrix_snapshots_step, t_run, rhomin, rhomax, N_e, N_i):
	import os
	import numpy as np
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	from brian2 import colorbar

	files_list = sorted(os.listdir(path_rho))

	# Generate list of timepoints of snapshots
	t = np.arange(0, t_run, rho_matrix_snapshots_step)

	# General plotting settings
	lwdth = 1.5
	s1 = 30
	s2 = 45
	mpl.rcParams['axes.linewidth'] = lwdth

	# Plotting
	counter = 0

	for i in files_list:
		print ('Plotting iteration rho E_E: ', str(counter), '/', len(t))

		data = np.loadtxt(os.path.join(path_rho, i), delimiter = ',')

		plt.close('all')

		fig = plt.figure(figsize = (10, 7.5))
		ax0 = fig.add_subplot(1, 1, 1) 

		plt.gcf().text(0, 1.02, 't = %s s' %(t[counter]), fontsize = s2, fontweight = 'bold')

		cmap = plt.get_cmap('gray')

		plt.pcolormesh(data.transpose(), cmap = cmap, vmin = rhomin, vmax = rhomax)

		cmap.set_under('black') 

		plt.axis([0, N_e, 0, N_e])
		plt.xlabel('Source neuron $E$', size = s1, labelpad = 15)
		plt.ylabel('Target neuron $E$', size = s1, labelpad = 15)
		plt.xticks(size = s1)
		plt.yticks(size = s1) 

		ax0.set_xticks(np.arange(0, N_e + 0.5, N_i))
		ax0.set_yticks(np.arange(0, N_e + 0.5, N_i))

		plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 10)

		cb = colorbar(pad = 0.05) 
		cb.set_label('Efficacy $\\rho$ (a.u.)', size = s1, labelpad = 15)
		cb.ax.tick_params(width = lwdth, labelsize = s1, pad = 10)

		plt.savefig(
			os.path.join(
			path_rho, sim_id + '_rho_snaps_E_E_'  + format(counter,'02') + '_time_' + format(t[counter], '05.2f') + 's.png'), 
			bbox_inches = 'tight')

		counter += 1

		plt.close()