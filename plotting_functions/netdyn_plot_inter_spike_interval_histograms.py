# -*- coding: utf-8 -*-
"""
@author: slehfeldt

Input:
- cwd: string holding the path of the current working directory
- full_path: string holding path to simulation folder
- sim_id: string holding the simulation identifier ('YYYYMMDD_hh_mm_ss')
- var_range: list with variable range of experiment
- isi: list of dictionaries of inter spike intervals returened by netdyn_inter_spike_intervals()
- exp_type: string describing the experiment 
- flag_isi: string describing the type of histogram to be plotted options are 'count' and 'percentage' flag_savefig: string according to exp_type for loop values in figurename

Output: 

Comments:
- Plot the inter spike interval histograms of the netdyn experiments.
"""
import os,sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from brian2 import mV, Hz

helper_dir = 'helper_functions'

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))

# User define modules
from inter_spike_interval_histograms import *

def netdyn_plot_inter_spike_interval_histograms(full_path, sim_id, var_range, isi, exp_type, flag_isi, flag_savefig):
	lwdth = 5.5
	s1 = 30
	s2 = 50

	mpl.rcParams['axes.linewidth'] = lwdth    

	for var_outer, freq in enumerate(var_range[0]):
		for var_inner, weight in enumerate(var_range[1]):
			# Get inter spike interval histograms
			bin_num = 25

			[ISI_hist_count_input_e, 
			edgs_input_e, 
			b_wdth_input_e, 
			ISI_hist_percentage_input_e] = inter_spike_interval_histograms(
				isi[0][var_outer, var_inner],
				bin_num)

			[ISI_hist_count_input_i, 
			edgs_input_i, 
			b_wdth_input_i, 
			ISI_hist_percentage_input_i] = inter_spike_interval_histograms(
				isi[1][var_outer, var_inner], 
				bin_num)

			[ISI_hist_count_e, 
			edgs_e, 
			b_wdth_e, 
			ISI_hist_percentage_e] = inter_spike_interval_histograms(
				isi[2][var_outer, var_inner],
				bin_num)

			[ISI_hist_count_i, 
			edgs_i, 
			b_wdth_i, 
			ISI_hist_percentage_i] = inter_spike_interval_histograms(
				isi[3][var_outer, var_inner],
				bin_num)

			# Plot inter spike interval histograms
			if flag_isi == 'count':
				fig  =  plt.figure(figsize = (30, 20))

				ax0 = fig.add_subplot(2, 2, 1)

				# Input population
				plt.bar(edgs_input_e[:-1], ISI_hist_count_input_e, b_wdth_input_e, edgecolor = 'grey', color = 'white', linewidth = lwdth/2)

				plt.xlabel('ISI Input_to_E (s)', size = s1, labelpad = 7)
				plt.ylabel('Count', size = s1, labelpad = 15)

				plt.xticks(size = s1, rotation = 40)
				plt.yticks(size = s1)

				plt.xlim(0, max(edgs_input_e))
				plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 10)

				ax1 = fig.add_subplot(2, 2, 2)

				plt.bar(edgs_input_i[:-1], ISI_hist_count_input_i, b_wdth_input_i, edgecolor = 'grey', color = 'white', linewidth = lwdth/2)

				plt.xlabel('ISI Input_to_I (s)', size = s1, labelpad = 7)
				plt.ylabel('Count', size = s1, labelpad = 15)

				plt.xticks(size = s1, rotation = 40)
				plt.yticks(size = s1)

				plt.xlim(0, max(edgs_input_e))

				plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 10)

				ax2 = fig.add_subplot(2, 2, 3)

				# Excitatory population
				plt.bar(edgs_e[:-1], ISI_hist_count_e, b_wdth_e, edgecolor = 'darkorange', color = 'white', linewidth = lwdth/2)

				plt.xlim(0, max(edgs_e))

				plt.xlabel('ISI E (s)', size = s1, labelpad = 7)
				plt.ylabel('Count', size = s1, labelpad = 15)

				plt.xticks(size = s1)
				plt.yticks(size = s1)
				plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 10)

				ax3 = fig.add_subplot(2, 2, 4)

				# Inhibitory population
				plt.bar(edgs_i[:-1], ISI_hist_count_i, b_wdth_i, edgecolor = 'royalblue', color = 'white', linewidth = lwdth/2)

				plt.xlim(0, max(edgs_i))

				plt.xlabel('ISI I (s)', size = s1, labelpad = 7)
				plt.ylabel('Count', size = s1, labelpad = 15)

				plt.xticks(size = s1)
				plt.yticks(size = s1)

				plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 10)

				plt.tight_layout()

				plt.savefig(
					os.path.join(full_path, sim_id + '_' + exp_type + \
						'_ISI_hist_count_stim_freq_' + \
						format(var_range[0][var_outer]/Hz, '08.3f') + \
						'_Hz_weight_' + format(var_range[1][var_inner]/mV, '08.3f') + '_mV.png'), 
					bbox_inches = 'tight')

			if flag_isi == 'percentage':
				fig  =  plt.figure(figsize = (30, 20))

				ax0 = fig.add_subplot(2, 2, 1)

				# Input_to_E population
				plt.bar(edgs_input_e[:-1], ISI_hist_percentage_input_e, b_wdth_input_e, edgecolor = 'grey', color = 'white', linewidth = lwdth/2)  

				plt.xlabel('ISI Input_to_E (s)', size = s1, labelpad = 7)
				plt.ylabel('Probability', size = s1, labelpad = 15)

				plt.xticks(size = s1, rotation = 40)
				plt.yticks(size = s1)

				plt.xlim(0, max(edgs_input_e))

				plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 10)

				# Input_to_I population
				ax1 = fig.add_subplot(2, 2, 2)

				plt.bar(edgs_input_i[:-1], ISI_hist_percentage_input_i, b_wdth_input_i, edgecolor = 'grey', color = 'white', linewidth = lwdth/2)

				plt.xlabel('ISI Input_to_I (s)', size = s1, labelpad = 7)
				plt.ylabel('Probability', size = s1, labelpad = 15)

				plt.xticks(size = s1, rotation = 40)
				plt.yticks(size = s1)

				plt.xlim(0, max(edgs_input_e))

				plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 10)

				ax2 = fig.add_subplot(2, 2, 3)

				# Excitatory population
				plt.bar(edgs_e[:-1], ISI_hist_percentage_e, b_wdth_e, edgecolor = 'darkorange', color = 'white', linewidth = lwdth/2)

				plt.xlim(0, max(edgs_e))

				plt.xlabel('ISI E (s)', size = s1, labelpad = 7)
				plt.ylabel('Probability', size = s1, labelpad = 15)

				plt.xticks(size = s1)
				plt.yticks(size = s1)

				plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 10)

				ax3 = fig.add_subplot(2, 2, 4)

				# Inhibitory population
				plt.bar(edgs_i[:-1], ISI_hist_percentage_i, b_wdth_i, edgecolor = 'royalblue', color = 'white', linewidth = lwdth/2)

				plt.xlim(0, max(edgs_i))

				plt.xlabel('ISI I (s)', size = s1, labelpad = 7)
				plt.ylabel('Probability', size = s1, labelpad = 15)

				plt.xticks(size = s1)
				plt.yticks(size = s1)

				plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 10)

				plt.tight_layout()

				if flag_savefig == 'freq_weight':
					plt.savefig(
						os.path.join(full_path, sim_id + '_' + exp_type + '_ISI_hist_percentage_stim_freq_' + format(var_range[0][var_outer]/Hz, '08.3f') + '_Hz_weight_' + format(var_range[1][var_inner]/mV, '08.3f') + '_mV.png'), 
						bbox_inches = 'tight')

				if flag_savefig == 'weight_weight':
					plt.savefig(
						os.path.join(full_path, sim_id + '_' + exp_type + '_ISI_hist_percentage_w1_' + format(var_range[0][var_outer]/mV, '08.3f') + '_mV_w2_' + format(var_range[1][var_inner]/mV, '08.3f') + '_mV.png'),
						bbox_inches = 'tight')
