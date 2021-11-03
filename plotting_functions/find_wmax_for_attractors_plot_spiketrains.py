# -*- coding: utf-8 -*-
"""
@author: slehfeldt

Inputs:

Outputs:

Comments:
- Plots the spike trains underlying the classification mechanism.
"""

def find_wmax_for_attractors_plot_spiketrains(sim_id, path_sim, s_tpoints_input_e, n_inds_input_e, s_tpoints_e, n_inds_e, tpoints_temp_p1, inds_temp_p1, tpoints_temp_p2, inds_temp_p2, t_run, t_start, t_end, stim_pulse_duration):

	import os
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	import numpy as np
	from brian2 import second

	# Go to simulation folder
	os.chdir(path_sim)  

	# General plotting settings
	
	lwdth = 2
	s1 = 35
	s2 = 75
	mpl.rcParams['axes.linewidth'] = lwdth

	plt.close('all')

	# E p1 full
	fig = plt.figure(figsize=(25, 35))
	ax1 = fig.add_subplot(3, 2, 1)

	plt.gcf().text(0.125, 0.92, 'a i)', fontsize = s2, fontweight = 'bold')
	plt.plot(s_tpoints_e,n_inds_e, '.', color = 'darkorange')

	plt.ylabel('Source neuron $E$', size = s1, labelpad = 35, horizontalalignment = 'center')

	ax1.set_yticks(np.arange(0, 257, 64))
	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 5, pad = 10)

	plt.ylim(0, 256)
	plt.yticks(size = s1)
	plt.xticks(size = s1)
	plt.xlim(stim_pulse_duration/second-1, stim_pulse_duration/second+1)

	plt.axvline(x = t_start[0]/second, ymin =  0, ymax = 256, color = 'k', linewidth = lwdth)
	plt.axvline(x = t_end[0]/second, ymin =  0, ymax = 256, color = 'k', linewidth = lwdth)

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 5, pad = 10)

	plt.xlabel('Time (s)', size = s1)

	# E p2 full
	ax2 = fig.add_subplot(3, 2, 2)
	plt.gcf().text(0.545, 0.92, 'a ii)', fontsize=s2, fontweight='bold')
	plt.plot(s_tpoints_e,n_inds_e, '.', color = 'darkorange')
	ax2.set_yticks(np.arange(0, 257, 64))

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 5, pad = 10)
	plt.ylim(0, 256)
	plt.yticks(size = s1)
	plt.xticks(size = s1)

	ax2.set_yticklabels([])

	plt.xlim(t_run/second-2, t_run/second)
	plt.axvline(x = t_start[1]/second, ymin = 0, ymax = 256, color = 'k', linewidth = lwdth)
	plt.axvline(x = t_end[1]/second, ymin = 0, ymax = 256, color = 'k', linewidth = lwdth)

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 5, pad = 10)
	plt.xlabel('Time (s)', size = s1)

	# E p1 extracted data
	ax1b = fig.add_subplot(3, 2, 3)

	plt.gcf().text(0.125, 0.622, 'a iii)', fontsize = s2, fontweight = 'bold')
	plt.plot(tpoints_temp_p1,inds_temp_p1, '.', color = 'darkorange')
	plt.ylabel('Source neuron $E$', size = s1, labelpad = 35, horizontalalignment = 'center')

	ax1b.set_yticks(np.arange(0, 257, 64))

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 5, pad = 10)

	plt.ylim(0, 256)
	plt.yticks(size = s1)
	plt.xticks(size = s1)
	plt.xlim(t_start[0]/second-0.05, t_end[0]/second+0.05)

	plt.axvline(x = t_start[0]/second, ymin =  0, ymax = 256, color = 'k', linewidth = lwdth)
	plt.axvline(x = t_end[0]/second, ymin =  0, ymax = 256, color = 'k', linewidth = lwdth)

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 5, pad = 10)
	plt.xlabel('Time (s)', size = s1)

	# E p2 extracted data
	ax2b = fig.add_subplot(3,2,4)
	plt.gcf().text(0.545, 0.622, 'a iv)', fontsize = s2, fontweight = 'bold')
	plt.plot(tpoints_temp_p2,inds_temp_p2, '.', color = 'darkorange')
	ax2b.set_yticks(np.arange(0, 257, 64))

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 5, pad = 10)
	plt.ylim(0, 256)
	plt.yticks(size = s1)
	plt.xticks(size = s1)

	ax2b.set_yticklabels([])
	plt.xlim(t_start[1]/second-0.1, t_end[1]/second)

	plt.axvline(x = t_start[1]/second, ymin =  0, ymax = 256, color = 'k', linewidth = lwdth)
	plt.axvline(x = t_end[1]/second, ymin =  0, ymax = 256, color = 'k', linewidth = lwdth)

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 5, pad = 10)
	plt.xlabel('Time (s)', size = s1)

	# Input_E p1 full
	ax3 = fig.add_subplot(3, 2, 5)
	plt.gcf().text(0.125, 0.326, 'b i)', fontsize = s2, fontweight = 'bold')
	plt.plot(s_tpoints_input_e,n_inds_input_e, '.', color = 'grey')

	plt.ylabel('Source neuron $Input E$', size = s1, labelpad = 35, horizontalalignment = 'center')

	ax3.set_yticks(np.arange(0, 257, 64))
	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 5, pad = 10)

	plt.ylim(0, 256)
	plt.yticks(size = s1)
	plt.xticks(size = s1)
	plt.xlim(stim_pulse_duration/second-1, stim_pulse_duration/second+1)

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 5, pad = 10)
	plt.axvline(x = t_start[0]/second, ymin =  0, ymax = 256, color = 'k')
	plt.axvline(x = t_end[0]/second, ymin =  0, ymax = 256, color = 'k')

	plt.xlabel('Time (s)', size = s1)

	# Input_E p2 full 
	ax4 = fig.add_subplot(3, 2, 6)
	plt.gcf().text(0.545, 0.326, 'b ii)', fontsize = s2, fontweight = 'bold')
	plt.plot(s_tpoints_input_e,n_inds_input_e, '.', color = 'grey')

	ax4.set_yticks(np.arange(0, 257, 64))
	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 5, pad = 10)

	plt.ylim(0, 256)
	plt.yticks(size = s1)
	plt.xticks(size = s1)
	plt.xlim(t_run/second-2, t_run/second)

	ax4.set_yticklabels([])

	plt.axvline(x = t_start[1]/second, ymin =  0, ymax = 256, color = 'k', linewidth = lwdth)
	plt.axvline(x = t_end[1]/second, ymin =  0, ymax = 256, color = 'k', linewidth = lwdth)

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 5, pad = 10)
	plt.xlabel('Time (s)', size = s1)

	plt.subplots_adjust(hspace = 0.4)

	plt.savefig(sim_id + '_spiketrains_classification.png', bbox_inches = 'tight')

	plt.close(fig)