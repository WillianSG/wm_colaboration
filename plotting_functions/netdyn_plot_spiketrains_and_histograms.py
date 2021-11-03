# -*- coding: utf-8 -*-
"""
@author: slehfeldt

Input:
- cwd: string holding the path of the current working directory
- full_path: string holding path to simulation folder
- sim_id: string holding the simulation identifier ('YYYYMMDD_hh_mm_ss')
- N: list with population sizes
- stim_size = stimulus size if other than N_pop
- var_range: list with variable range of experiment
- s_tpoints: list with dictionaries of spike times of each population
- n_inds: list with dictionaries of neuron indices of each population
- t_run: duration of simulation
- bin_width_desired: desired bin width of time_resolved histogram 
- for tpoints without units: bin_width in seconds and also without unit
- for tpoints with units: bin_width with unit
- exp_type: string describing the experiment
- flag_savefig: string according to exp_type for loop values in figurename

Output: 

Comments:
- Creates plots of the populations' spiketrains and firing rate histograms ('neuron-resolved' and 'time-resolved') and stores the figures into the simulation folder.
"""

import os, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from brian2 import mV, Hz, second, ms, mean, std

helper_dir = 'helper_functions'

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))

# User define modules
from firing_rate_histograms import *

def netdyn_plot_spiketrains_and_histograms(full_path, sim_id, N, stim_size,var_range, s_tpoints, n_inds, t_run, bin_width_desired, exp_type, flag_savefig):

	# General style settings
	lwdth=2.5
	s1=30
	s2=70
	mpl.rcParams['axes.linewidth'] = lwdth

	for var_outer, freq in enumerate(var_range[0]):
		for var_inner, weight in enumerate(var_range[1]):
			# 1 - Resolving histograms

			# 1.1 - 'neuron_resolved' histograms

			[input_e_n_hist_count, 
			input_e_n_hist_edgs,
			input_e_n_hist_fr] = firing_rate_histograms(
				tpoints = s_tpoints[0][var_outer,var_inner],
				inds = n_inds[0][var_outer,var_inner],
				bin_width = None,
				N_pop = stim_size,
				flag_hist = 'neuron_resolved')

			[input_i_n_hist_count, 
			input_i_n_hist_edgs,
			input_i_n_hist_fr] = firing_rate_histograms(
				tpoints = s_tpoints[1][var_outer,var_inner],
				inds = n_inds[1][var_outer,var_inner],
				bin_width = None,
				N_pop = N[1], 
				flag_hist = 'neuron_resolved')

			[e_n_hist_count, 
			e_n_hist_edgs, 
			e_n_hist_fr] = firing_rate_histograms(
				tpoints = s_tpoints[2][var_outer,var_inner], 
				inds = n_inds[2][var_outer,var_inner],
				bin_width = None, 
				N_pop = stim_size, 
				flag_hist = 'neuron_resolved')

			[i_n_hist_count, 
			i_n_hist_edgs,
			i_n_hist_fr] = firing_rate_histograms(
				tpoints = s_tpoints[3][var_outer,var_inner], 
				inds = n_inds[3][var_outer,var_inner],
				bin_width = None, 
				N_pop = N[3], 
				flag_hist = 'neuron_resolved')

			# 1.2 - 'time_resolved' histograms

			[input_e_t_hist_count, 
			input_e_t_hist_edgs,
			input_e_t_hist_bin_widths,
			input_e_t_hist_fr] = firing_rate_histograms(
				tpoints = s_tpoints[0][var_outer,var_inner], 
				inds = n_inds[0][var_outer,var_inner],
				bin_width = bin_width_desired,
				N_pop = stim_size, 
				flag_hist = 'time_resolved')

			[input_i_t_hist_count, 
			input_i_t_hist_edgs,
			input_i_t_hist_bin_widths,
			input_i_t_hist_fr] = firing_rate_histograms(
				tpoints = s_tpoints[1][var_outer,var_inner], 
				inds = n_inds[1][var_outer,var_inner],
				bin_width = bin_width_desired,
				N_pop = N[1], 
				flag_hist = 'time_resolved')

			[e_t_hist_count, 
			e_t_hist_edgs, 
			e_t_hist_bin_widths,
			e_t_hist_fr] = firing_rate_histograms(
				tpoints = s_tpoints[2][var_outer,var_inner], 
				inds = n_inds[2][var_outer,var_inner],
				bin_width = bin_width_desired, 
				N_pop = stim_size, 
				flag_hist = 'time_resolved')

			[i_t_hist_count, 
			i_t_hist_edgs,
			i_t_hist_bin_widths,
			i_t_hist_fr] = firing_rate_histograms(
				tpoints = s_tpoints[3][var_outer,var_inner], 
				inds = n_inds[3][var_outer,var_inner],
				bin_width = bin_width_desired,
				N_pop = N[3], 
				flag_hist = 'time_resolved')                                              

			# 2 - Calculate mean values and standard deviations of histograms

			# 2.1 - 'neuron_resolved' histograms
			n_hist_rate_mean_input_e = np.mean(input_e_n_hist_fr) 
			n_hist_rate_std_input_e = np.std(input_e_n_hist_fr)
			n_hist_rate_mean_input_i = np.mean(input_i_n_hist_fr) 
			n_hist_rate_std_input_i = np.std(input_i_n_hist_fr)
			n_hist_rate_mean_e = np.mean(e_n_hist_fr) 
			n_hist_rate_std_e = np.std(e_n_hist_fr)
			n_hist_rate_mean_i = np.mean(i_n_hist_fr) 
			n_hist_rate_std_i = np.std(i_n_hist_fr)

			# 2.2 - 'time_resolved' histograms
			t_hist_rate_mean_input_e = np.mean(input_e_t_hist_fr)  
			t_hist_rate_std_input_e = np.std(input_e_t_hist_fr)
			t_hist_rate_mean_input_i = np.mean(input_i_t_hist_fr) 
			t_hist_rate_std_input_i = np.std(input_i_t_hist_fr)
			t_hist_rate_mean_e = np.mean(e_t_hist_fr)  
			t_hist_rate_std_e = np.std(e_t_hist_fr)
			t_hist_rate_mean_i = np.mean(i_t_hist_fr) 
			t_hist_rate_std_i = np.std(i_t_hist_fr)

			# 3 - Plotting

			plt.close("all")

			fig = plt.figure(figsize = (25, 35))

			gs = gridspec.GridSpec(10, 2, height_ratios = [1, 4, 1, 1, 1, 4, 1, 1, 1, 1], width_ratios = [5,1])
			gs.update(wspace = 0.07, hspace = 0.3, left = None, right = None)

			# Inhibitory population: spike trains
			ax0 = fig.add_subplot(gs[0,0])

			plt.plot(s_tpoints[3][var_outer, var_inner], n_inds[3][var_outer,var_inner], '.', color = 'green')
			plt.ylim(0, N[3])
			plt.xlim(0, t_run/second)
			plt.tick_params(which = 'major', width = lwdth, length = 9,labelsize = s1)

			ax0.set_yticks(np.arange(0, N[3]+0.5, N[3]/2))
			ax0.tick_params(axis = 'y', which = 'major', pad = 15)
			ax0.set_xticklabels([])

			plt.ylabel('Source\nneuron $I$', size = s1, labelpad = 35, horizontalalignment = 'center')

			# Inhibitory population: firing rate historgram 'neuron_resolved'
			ax0b = fig.add_subplot(gs[0, -1])

			plt.barh(i_n_hist_edgs[:-1], i_n_hist_fr, edgecolor = 'green', color = 'white', linewidth = lwdth/2)

			ax0b.xaxis.set_tick_params(labeltop = 'on', labelbottom = 'off')
			ax0b.set_yticklabels([])
			ax0b.set_yticks(np.arange(0, N[3]+0.5, N[3]/2))

			plt.tick_params(which = 'major', width = lwdth, length = 9,labelsize = s1, pad = 15)
			plt.axvline(x = n_hist_rate_mean_i, ymin = 0, ymax = N[3], linewidth = lwdth/2, color = 'grey')

			ax0b.axvspan(n_hist_rate_mean_i-n_hist_rate_std_i,n_hist_rate_mean_i+n_hist_rate_std_i, alpha = 0.5, color = 'grey')

			plt.ylim(0, N[3])

			ax0b.set_xticks([n_hist_rate_mean_i])
			ax0b.set_xticklabels([round(n_hist_rate_mean_i, 1)])

			# Excitatory population: spike trains
			ax1 = fig.add_subplot(gs[1,0])

			plt.plot(s_tpoints[2][var_outer, var_inner], n_inds[2][var_outer,var_inner], '.', color = 'mediumblue')
			plt.tick_params(which = 'major',width = lwdth,length = 9,labelsize = s1,pad = 15)

			ax1.set_yticks(np.arange(0, N[2]+0.5, N[3]))
			ax1.tick_params(axis = 'y', which = 'major', pad = 15)
			ax1.set_xticklabels([])

			plt.ylabel('Source neuron $E$', size = s1, labelpad = 35, horizontalalignment = 'center')
			plt.ylim(0, N[2])
			plt.xlim(0, t_run/second)

			# Excitatory population: firing rate historgram 'neuron_resolved'
			ax1b = fig.add_subplot(gs[1, -1])

			plt.barh(e_n_hist_edgs[:-1],e_n_hist_fr,edgecolor = 'mediumblue',color = 'white', linewidth = lwdth/2)

			ax1b.set_yticklabels([])
			ax1b.set_yticks(np.arange(0, N[2]+0.5, N[3]))

			plt.tick_params(which = 'major', width = lwdth, length = 9,labelsize = s1, pad = 15)

			plt.axvline(x = n_hist_rate_mean_e, ymin = 0, ymax = N[1], linewidth = lwdth/2, color = 'grey')

			ax1b.axvspan(n_hist_rate_mean_e-n_hist_rate_std_e,n_hist_rate_mean_e+n_hist_rate_std_e, alpha = 0.5, color = 'grey')

			plt.ylim(0,N[2])

			ax1b.set_xticks([n_hist_rate_mean_e])
			ax1b.set_xticklabels([round(n_hist_rate_mean_e, 1)])

			plt.xlabel('$FR$ (Hz)', size=  s1, labelpad=  15)

			# Inhibitory population: firing rate histogram 'time_resolved'
			ax2 = fig.add_subplot(gs[2,0])

			plt.bar(i_t_hist_edgs[:-1],i_t_hist_fr,i_t_hist_bin_widths,edgecolor = 'green',color = 'white',linewidth = lwdth/2)
			plt.axhline(y = t_hist_rate_mean_i,xmin = 0,xmax = t_run/ms,linewidth = lwdth/2,color = 'grey')

			ax2.axhspan(t_hist_rate_mean_i-t_hist_rate_std_i,t_hist_rate_mean_i+t_hist_rate_std_i,alpha = 0.5,color = 'grey')

			plt.xlim(0,t_run/second)

			ax2.set_yticks([t_hist_rate_mean_i])
			ax2.set_yticklabels([round(t_hist_rate_mean_i,1)])        
			ax2.tick_params(axis = 'y', which = 'major', pad = 15)

			plt.tick_params(which = 'major', width = lwdth, length = 9,labelsize = s1)

			ax2.set_xticklabels([])

			plt.ylabel('$FR_{I}$\n(Hz)', size = s1, labelpad = 35, horizontalalignment = 'center')

			# Excitatory population: firing rate histogram 'time_resolved'
			ax3 = fig.add_subplot(gs[3,0])

			plt.bar(e_t_hist_edgs[:-1],e_t_hist_fr,e_t_hist_bin_widths,edgecolor = 'mediumblue',color = 'white',linewidth = lwdth/2)
			plt.axhline(y = t_hist_rate_mean_e,xmin = 0,xmax = t_run/ms,linewidth = lwdth/2,color = 'grey')

			ax3.axhspan(t_hist_rate_mean_e-t_hist_rate_std_e,t_hist_rate_mean_e+t_hist_rate_std_e,alpha = 0.5,color = 'grey')

			plt.xlim(0, t_run/second)

			ax3.set_yticks([t_hist_rate_mean_e])
			ax3.set_yticklabels([round(t_hist_rate_mean_e, 1)])

			plt.tick_params(which = 'major', width = lwdth, length = 9,labelsize = s1, pad = 15)

			plt.xlabel('Time (s)', size = s1, labelpad = 20)
			plt.ylabel('$FR_{E}$\n(Hz)', size = s1, labelpad = 35, horizontalalignment = 'center')

			# Input_E population: spike trains
			ax4 = fig.add_subplot(gs[5,0])

			plt.plot(s_tpoints[0][var_outer,var_inner], n_inds[0][var_outer,var_inner], '.', color = 'grey')

			plt.ylim(0, N[0])
			plt.xlim(0, t_run/second)

			plt.tick_params(which = 'major',width = lwdth,length = 9,labelsize = s1)
			ax4.set_yticks(np.arange(0, N[0]+0.5, N[3]))

			ax4.tick_params(axis = 'y', which = 'major', pad = 15)
			ax4.set_xticklabels([])

			plt.ylabel('Source neuron\n$Input E$', size = s1, labelpad = 35, horizontalalignment = 'center')

			# Input_E population: firing rate historgram 'neuron_resolved'
			ax4b = fig.add_subplot(gs[5,-1])

			plt.barh(input_e_n_hist_edgs[:-1], input_e_n_hist_fr, edgecolor = 'grey',color = 'white',linewidth = lwdth/2)

			ax4b.set_yticklabels([])
			ax4b.set_yticks(np.arange(0, N[0]+0.5, N[3]))

			plt.tick_params(which = 'major', width = lwdth, length = 9,labelsize = s1, pad = 15)

			plt.axvline(x = n_hist_rate_mean_input_e, ymin = 0, ymax = N[0],linewidth = lwdth/2, color = 'grey')

			ax4b.axvspan(n_hist_rate_mean_input_e-n_hist_rate_std_input_e,n_hist_rate_mean_input_e+n_hist_rate_std_input_e, alpha = 0.5,color = 'grey')

			plt.ylim(0, N[0])

			ax4b.set_xticks([n_hist_rate_mean_input_e])
			ax4b.set_xticklabels([round(n_hist_rate_mean_input_e, 1)])

			plt.xlabel('$FR_{Input E}$ (Hz)', size = s1, labelpad = 15)

			# Input_E population: firing rate histogram 'time_resolved'
			ax5 = fig.add_subplot(gs[6,0])

			plt.bar(input_e_t_hist_edgs[:-1],input_e_t_hist_fr,input_e_t_hist_bin_widths, edgecolor = 'grey', color = 'white',linewidth = lwdth/2)
			plt.axhline(y = t_hist_rate_mean_input_e, xmin = 0, xmax = t_run/ms, linewidth = lwdth/2, color = 'grey')

			ax5.axhspan(t_hist_rate_mean_input_e-t_hist_rate_std_input_e,t_hist_rate_mean_input_e+t_hist_rate_std_input_e, alpha = 0.5,color = 'grey')

			plt.xlim(0, t_run/second)

			ax5.set_yticks([t_hist_rate_mean_input_e])
			ax5.set_yticklabels([round(t_hist_rate_mean_input_e,1)])     
			ax5.tick_params(axis = 'y', which = 'major', pad = 15)

			plt.tick_params(which = 'major',width = lwdth,length = 9,labelsize = s1,pad = 15)
			plt.xlabel('Time (s)', size = s1, labelpad = 20)
			plt.ylabel('$FR_{Input E}$\n(Hz)', size = s1, labelpad = 15)

			# Input_I population: spike trains
			ax4 = fig.add_subplot(gs[8,0])

			plt.plot(s_tpoints[1][var_outer,var_inner], n_inds[1][var_outer,var_inner], '.', color = 'grey')

			plt.ylim(0, N[1])
			plt.xlim(0, t_run/second)

			plt.tick_params(which = 'major',width = lwdth,length = 9,labelsize = s1)

			ax4.set_yticks(np.arange(0, N[1]+0.5, N[3]/2))
			ax4.tick_params(axis = 'y', which = 'major', pad = 15)
			ax4.set_xticklabels([])

			plt.ylabel('Source\nneuron\n$Input I$', size = s1, labelpad = 35, horizontalalignment = 'center')

			# Input_I population: firing rate historgram 'neuron_resolved'
			ax4b = fig.add_subplot(gs[8,-1])

			plt.barh(input_i_n_hist_edgs[:-1], input_i_n_hist_fr, edgecolor = 'grey', color = 'white', linewidth = lwdth/2)

			ax4b.set_yticklabels([])
			ax4b.set_yticks(np.arange(0, N[1]+0.5, N[3]))

			plt.tick_params(which = 'major', width = lwdth, length = 9,labelsize = s1, pad = 15)

			plt.axvline(x = n_hist_rate_mean_input_i, ymin = 0, ymax = N[0],linewidth = lwdth/2, color = 'grey')

			ax4b.axvspan(n_hist_rate_mean_input_i-n_hist_rate_std_input_i,n_hist_rate_mean_input_i+n_hist_rate_std_input_i, alpha = 0.5,color = 'grey')

			plt.ylim(0, N[1])

			ax4b.set_xticks([n_hist_rate_mean_input_i])
			ax4b.set_xticklabels([round(n_hist_rate_mean_input_i,1)])

			plt.xlabel('$FR_{Input I}$ (Hz)',size = s1,labelpad = 15)

			# Input_I population: firing rate histogram 'time_resolved'
			ax5 = fig.add_subplot(gs[9,0])

			plt.bar(input_i_t_hist_edgs[:-1],input_i_t_hist_fr,input_i_t_hist_bin_widths,edgecolor='grey',color='white',linewidth=lwdth/2)
			plt.axhline(y=t_hist_rate_mean_input_i,xmin=0,xmax=t_run/ms,linewidth=lwdth/2,color='grey')

			ax5.axhspan(t_hist_rate_mean_input_i-t_hist_rate_std_input_i,t_hist_rate_mean_input_i+t_hist_rate_std_input_i,alpha=0.5,color='grey')

			plt.xlim(0,t_run/second)

			ax5.set_yticks([t_hist_rate_mean_input_i])
			ax5.set_yticklabels([round(t_hist_rate_mean_input_i,1)])       
			ax5.tick_params(axis = 'y', which = 'major', pad = 15)

			plt.tick_params(which = 'major', width = lwdth, length = 9,labelsize = s1, pad = 15)
			plt.xlabel('Time (s)', size = s1, labelpad = 20)
			plt.ylabel('$FR_{Input I}$\n(Hz)', size = s1, labelpad = 15)

			if flag_savefig == 'freq_weight':
				plt.savefig(os.path.join(full_path, sim_id + '_' + exp_type + '_stim_freq_' + format(var_range[0][var_outer]/Hz, '08.3f') + '_Hz_weight_' + format(var_range[1][var_inner]/mV, '08.3f') + '_mV.png'), bbox_inches = 'tight')

				plt.close(fig)

			if flag_savefig == 'weight_weight':
				plt.savefig(os.path.join(full_path, sim_id + '_' + exp_type + '_w1_' + format(var_range[0][var_outer]/mV, '08.3f') + '_mV_w2_' + format(var_range[1][var_inner]/mV, '08.3f') + '_mV.png'), bbox_inches = 'tight')

				plt.close(fig)


