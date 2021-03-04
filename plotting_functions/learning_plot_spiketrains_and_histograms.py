# -*- coding: utf-8 -*-
"""
@author: asonntag (adapted from Lehfeldt)

Inputs:
- path_sim: plot output destination

Outputs:

Comments:
"""

import os, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from brian2 import mV, Hz, second, ms, mean, std
from firing_rate_histograms import firing_rate_histograms

def learning_plot_spiketrains_and_histograms(
	sim_id, 
	path_sim, 
	stim_size, 
	N, 
	s_tpoints, 
	n_inds, 
	bin_width_desired, 
	t_run, 
	exp_type):

	# General plotting settings

	lwdth = 3
	s1 = 45
	s2 = 95
	mpl.rcParams['axes.linewidth'] = lwdth

	plt.close('all')

	# Calculating firing rate histograms for plotting

	# 'time_resolved' histograms - [?]
	[input_e_t_hist_count, 
	input_e_t_hist_edgs,
	input_e_t_hist_bin_widths,
	input_e_t_hist_fr] = firing_rate_histograms(
		tpoints = s_tpoints[0],
		inds = n_inds[0],
		bin_width = bin_width_desired,
		N_pop = stim_size,
		flag_hist = 'time_resolved')

	[input_i_t_hist_count,
	input_i_t_hist_edgs,
	input_i_t_hist_bin_widths,
	input_i_t_hist_fr] = firing_rate_histograms(
		tpoints = s_tpoints[1],
		inds = n_inds[1],
		bin_width = bin_width_desired,
		N_pop = N[1],
		flag_hist = 'time_resolved')

	[e_t_hist_count, 
	e_t_hist_edgs, 
	e_t_hist_bin_widths,
	e_t_hist_fr] = firing_rate_histograms(
		tpoints = s_tpoints[2],
		inds = n_inds[2],
		bin_width = bin_width_desired,
		N_pop = stim_size,
		flag_hist = 'time_resolved')

	[i_t_hist_count,
	i_t_hist_edgs,
	i_t_hist_bin_widths,
	i_t_hist_fr] = firing_rate_histograms(
		tpoints = s_tpoints[3],
		inds = n_inds[3],
		bin_width = bin_width_desired,
		N_pop = N[3],
		flag_hist = 'time_resolved')

	# Plotting spiking activity and histograms

	fig = plt.figure(figsize = (65, 45))

	gs = gridspec.GridSpec(9, 1, 
		height_ratios = [1, 0.5, 1, 1, 0.5, 4, 1, 0.5, 1])
	gs.update(wspace = 0.07, hspace = 0.3, left = None, right = None)

	# Input_I population: histogram
	ax0 = fig.add_subplot(gs[0, 0])

	plt.bar(input_i_t_hist_edgs[:-1], input_i_t_hist_fr, input_i_t_hist_bin_widths, edgecolor = 'grey', color = 'white', linewidth = lwdth)

	plt.ylabel('$\\nu_{Input I}$\n(Hz)', size = s1, labelpad = 35, 
		horizontalalignment = 'center')
	plt.xlim(0, t_run/second)
	plt.ylim(0, 5500)

	ax0.set_yticks(np.arange(0, 5510, 2500))
	ax0.set_xticklabels([])

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 10, pad = 10)
	plt.yticks(size = s1)
	plt.xticks(size = s1)

	# Inhibitory population: spiking
	ax1 = fig.add_subplot(gs[2, 0])

	plt.plot(s_tpoints[3], n_inds[3], '.', color = 'lightcoral')
	plt.ylabel('Source\nneuron $I$', size = s1, labelpad = 105, horizontalalignment = 'center')

	ax1.set_yticks(np.arange(0, N[3]+1, N[3]/2))
	ax1.set_xticklabels([])

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 10, pad = 10)
	plt.ylim(0,N[3])
	plt.yticks(size = s1)
	plt.xticks(size = s1)
	plt.xlim(0, t_run/second)

	# Inhibitory population: histogram
	ax2 = fig.add_subplot(gs[3, 0])

	plt.bar(i_t_hist_edgs[:-1], i_t_hist_fr, i_t_hist_bin_widths, edgecolor = 'lightcoral', color = 'white', linewidth = lwdth)
	plt.ylabel('$\\nu_{I}$\n(Hz)', size = s1, labelpad = 35, horizontalalignment = 'center')
	plt.xlim(0, t_run/second)
	plt.ylim(0, max(i_t_hist_fr)*1.1)

	yticks = np.linspace(0, max(i_t_hist_fr)*1.1, 3)

	ax2.set_yticks(yticks)
	ax2.set_yticklabels(np.around(yticks))
	ax2.set_xticklabels([])

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 10,
		pad = 10)
	plt.yticks(size = s1)
	plt.xticks(size = s1)

	# Spont Inhibitory population: spiking
	ax1 = fig.add_subplot(gs[2, 0])

	plt.plot(s_tpoints[3], n_inds[3], '.', color = 'lightcoral')
	plt.ylabel('Source\nneuron $I$', size = s1, labelpad = 105, horizontalalignment = 'center')

	ax1.set_yticks(np.arange(0, N[3]+1, N[3]/2))
	ax1.set_xticklabels([])

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 10, pad = 10)
	plt.ylim(0,N[3])
	plt.yticks(size = s1)
	plt.xticks(size = s1)
	plt.xlim(0, t_run/second)

	# Spont Inhibitory population: histogram
	ax2 = fig.add_subplot(gs[3, 0])

	plt.bar(i_t_hist_edgs[:-1], i_t_hist_fr, i_t_hist_bin_widths, edgecolor = 'lightcoral', color = 'white', linewidth = lwdth)
	plt.ylabel('$\\nu_{I}$\n(Hz)', size = s1, labelpad = 35, horizontalalignment = 'center')
	plt.xlim(0, t_run/second)
	plt.ylim(0, max(i_t_hist_fr)*1.1)

	yticks = np.linspace(0, max(i_t_hist_fr)*1.1, 3)

	ax2.set_yticks(yticks)
	ax2.set_yticklabels(np.around(yticks))
	ax2.set_xticklabels([])

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 10,
		pad = 10)
	plt.yticks(size = s1)
	plt.xticks(size = s1)

	# == Excitatory population: spiking
	ax3 = fig.add_subplot(gs[5, 0])

	plt.plot(s_tpoints[2],n_inds[2], '.', color = 'cornflowerblue')
	plt.ylabel('Source neuron $E$', size = s1, labelpad = 35, horizontalalignment = 'center')

	ax3.set_yticks(np.arange(0, N[2]+1, N[3]))
	ax3.set_xticklabels([])

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 10, pad = 10)
	plt.ylim(0, N[2])
	plt.yticks(size = s1)
	plt.xticks(size = s1)
	plt.xlim(0, t_run/second)

	# Excitatory population: histogram
	ax4 = fig.add_subplot(gs[6, 0]) 

	plt.bar(e_t_hist_edgs[:-1], e_t_hist_fr, e_t_hist_bin_widths, 
		edgecolor = 'cornflowerblue', color = 'white', linewidth = lwdth)
	plt.ylabel('$\\nu_{E}$\n(Hz)', size = s1, labelpad = 35,
		horizontalalignment = 'center')
	plt.xlim(0, t_run/second)

	yticks = np.linspace(0, max(e_t_hist_fr)*1.1, 3)

	ax4.set_yticks(yticks)
	ax4.set_yticklabels(np.around(yticks))
	ax4.set_xticklabels([])

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 10,
		pad = 10)
	plt.yticks(size = s1)
	plt.xticks(size = s1)

	# Input_E population: histogram
	ax5 = fig.add_subplot(gs[8, 0])

	plt.bar(input_e_t_hist_edgs[:-1], input_e_t_hist_fr,
		input_e_t_hist_bin_widths, 
		edgecolor = 'grey', 
		color = 'white', 
		linewidth = lwdth)
	plt.ylabel('$\\nu_{Input E}$\n(Hz)', size = s1, labelpad = 35, horizontalalignment = 'center')
	plt.xlim(0, t_run/second)
	plt.ylim(0, 4500)

	ax5.set_yticks(np.arange(0, 4500, 2000))

	plt.yticks(size = s1)
	plt.xticks(size = s1)
	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 10,
		pad = 15)
	plt.xlabel('Time (s)', size = s1)
	plt.savefig(path_sim + '_population_spiking.png', bbox_inches = 'tight')
	plt.close(fig)
