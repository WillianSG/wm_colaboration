# -*- coding: utf-8 -*-
"""
@author: slehfeldt

Input:
- cwd: string holding the path of the current working directory
- full_path: string holding path to simulation folder
- sim_id: string holding the simulation identifier ('YYYYMMDD_hh_mm_ss')
- var_range: list with variable range of experiment
- var_steps: list with step size of each running variable of experiment
- dicts_means: dictionary with mean firing rate of each loop iteration fo experiment
- dicts_stds: dictionary with standard deviations fo firing rates of each loop iteration
- exp_type: string describing the experiment  
- x/ylabel: string setting the figure x and y labels

Output: 

Comments:
- Plots the mean firing rates and standard deviations as colour grids over the applied value range of the experiment.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import numpy as np
from brian2 import colorbar

def netdyn_plot_colourgrid_mean_rates_and_stds(full_path, sim_id, var_range, var_steps, dicts_means, dicts_stds, exp_type, xlabel, ylabel):
	
	lwdth = 2.5
	s1 = 50
	s2 = 65
	mpl.rcParams['axes.linewidth'] = lwdth            
	colourmap_e = 'Blues'
	colourmap_i = 'Greens'
	colourmap_inputs = 'Greys'

	# Empty arrays for storage of mean firing rates
	array_rates_mean_input_e = np.zeros((len(var_range[0]),len(var_range[1])))
	array_rates_std_input_e = np.zeros((len(var_range[0]),len(var_range[1]))) 
	array_rates_mean_input_i = np.zeros((len(var_range[0]),len(var_range[1])))
	array_rates_std_input_i = np.zeros((len(var_range[0]),len(var_range[1]))) 
	array_rates_mean_e = np.zeros((len(var_range[0]),len(var_range[1])))
	array_rates_std_e = np.zeros((len(var_range[0]),len(var_range[1])))  
	array_rates_mean_i = np.zeros((len(var_range[0]),len(var_range[1])))
	array_rates_std_i = np.zeros((len(var_range[0]),len(var_range[1]))) 

	# Convert dictionaries to 2d array
	for var_outer in np.arange(0,len(var_range[0])):
		for var_inner in np.arange(0,len(var_range[1])): 
			array_rates_mean_input_e[var_outer, var_inner] = dicts_means[0][var_outer, var_inner]
			array_rates_std_input_e[var_outer, var_inner] = dicts_stds[0][var_outer, var_inner]

			array_rates_mean_input_i[var_outer, var_inner] = dicts_means[1][var_outer, var_inner]
			array_rates_std_input_i[var_outer, var_inner] = dicts_stds[1][var_outer, var_inner]

			array_rates_mean_e[var_outer, var_inner] = dicts_means[2][var_outer, var_inner]
			array_rates_std_e[var_outer, var_inner] = dicts_stds[2][var_outer, var_inner]

			array_rates_mean_i[var_outer, var_inner] = dicts_means[3][var_outer, var_inner]
			array_rates_std_i[var_outer, var_inner] = dicts_stds[3][var_outer, var_inner]

	# === Plot 1 - Colour grid of mean firing rates

	# Merge all rates of E & I into one list in order to find max --> color bar legend limits
	E_I_all_rates_concatenated = np.concatenate((array_rates_mean_e,array_rates_mean_i), axis = 0)
	E_I_all_rates_merged = list(itertools.chain.from_iterable(E_I_all_rates_concatenated))

	# Input_to_E population: mean firing rate
	plt.close("all")

	fig = plt.figure(figsize = (21, 21))
	ax1 = fig.add_subplot(1, 1, 1)

	plt.pcolor(array_rates_mean_input_e.transpose(), cmap = colourmap_inputs, vmin = 0, vmax = np.nanmax(array_rates_mean_input_e))

	ax1.set_xticks(np.arange(0.5, len(var_range[0]), 1))
	ax1.set_xticklabels(var_range[0], rotation = 40)
	ax1.set_yticks(np.arange(0.5, len(var_range[1]), 1))
	ax1.set_yticklabels(var_range[1])

	plt.xlabel(xlabel, size = s2, labelpad = 7)
	plt.ylabel(ylabel[0], size = s2, labelpad = 15)
	plt.xticks(size = s1)
	plt.yticks(size = s1)

	ax1.set_xlim(xmin = 0,  xmax = len(var_range[0]))
	ax1.set_ylim(ymin = 0,  ymax = len(var_range[1]))

	cb = colorbar() 
	cb.set_label('Mean $FR_{Input E}$ (Hz)',  size = s1,  labelpad = 35)
	cb.ax.tick_params(width = lwdth, labelsize = s1, pad = 10)

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 10)

	plt.savefig(os.path.join(full_path, sim_id + "_" + exp_type + "_rates_mean_colour_Input_to_E.png"), bbox_inches = 'tight')

	# Input_to_I population: mean firing rate
	plt.close("all")

	fig  =  plt.figure(figsize = (21, 21))
	ax1 = fig.add_subplot(1, 1, 1)

	plt.pcolor(array_rates_mean_input_i.transpose(), cmap = colourmap_inputs, vmin = 0, vmax = np.nanmax(array_rates_mean_input_i))

	ax1.set_xticks(np.arange(0.5, len(var_range[0]), 1))
	ax1.set_xticklabels(var_range[0], rotation = 40)
	ax1.set_yticks(np.arange(0.5, len(var_range[1]), 1))
	ax1.set_yticklabels(var_range[1])

	plt.xlabel(xlabel, size = s2, labelpad = 7)
	plt.ylabel(ylabel[1], size = s2, labelpad = 15)
	plt.xticks(size = s1)
	plt.yticks(size = s1)

	ax1.set_xlim(xmin = 0, xmax = len(var_range[0]))
	ax1.set_ylim(ymin = 0, ymax = len(var_range[1]))

	cb  =  colorbar() 
	cb.set_label('Mean $FR_{Input I}$ (Hz)', size = s1, labelpad = 35)
	cb.ax.tick_params(width = lwdth, labelsize = s1, pad = 10)

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 10)

	plt.savefig(os.path.join(full_path, sim_id + "_" + exp_type + "_rates_mean_colour_Input_to_I.png"), bbox_inches = 'tight')

	# Excitatory population: mean firing rate
	plt.close("all")

	fig  =  plt.figure(figsize = (21, 21))
	ax1 = fig.add_subplot(1, 1, 1)

	plt.pcolor(array_rates_mean_e.transpose(), cmap = colourmap_e, vmin = 0, vmax = np.nanmax(E_I_all_rates_merged))

	ax1.set_xticks(np.arange(0.5, len(var_range[0]), 2))
	ax1.set_xticklabels(var_range[0][::2], rotation = 40)
	ax1.set_yticks(np.arange(0.5, len(var_range[1]), 2))
	ax1.set_yticklabels(var_range[1][::2])

	plt.xlabel(xlabel, size = s2, labelpad = 7)
	plt.ylabel(ylabel[0], size = s2, labelpad = 15)
	plt.xticks(size = s1)
	plt.yticks(size = s1)

	ax1.set_xlim(xmin = 0, xmax = len(var_range[0]))
	ax1.set_ylim(ymin = 0, ymax = len(var_range[1]))

	cb  =  colorbar() 
	cb.set_label('Mean $FR_{E}$ (Hz)', size = s1, labelpad = 35)
	cb.ax.tick_params(width = lwdth, labelsize = s1, pad = 10)

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 10)

	plt.savefig(os.path.join(full_path, sim_id + "_" + exp_type + "_rates_mean_colour_E.png"), bbox_inches = 'tight')

	# Inhibitory population: mean firing rate
	plt.close("all")

	fig  =  plt.figure(figsize = (21, 21))
	ax1 = fig.add_subplot(1, 1, 1)
	plt.pcolor(array_rates_mean_i.transpose(), cmap = colourmap_i, vmin = 0, vmax = np.nanmax(E_I_all_rates_merged))

	ax1.set_xticks(np.arange(0.5, len(var_range[0]), 2))
	ax1.set_xticklabels(var_range[0][::2], rotation = 40)
	ax1.set_yticks(np.arange(0.5, len(var_range[1]), 2))
	ax1.set_yticklabels(var_range[1][::2])

	plt.xlabel(xlabel, size = s2, labelpad = 7)
	plt.ylabel(ylabel[1], size = s2, labelpad = 15)
	plt.xticks(size = s1)
	plt.yticks(size = s1)

	ax1.set_xlim(xmin = 0, xmax = len(var_range[0]))
	ax1.set_ylim(ymin = 0, ymax = len(var_range[1]))

	cb  =  colorbar() 
	cb.set_label('Mean $FR_{I}$ (Hz)', size = s1, labelpad = 35)
	cb.ax.tick_params(width = lwdth, labelsize = s1, pad = 10)

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 10)

	plt.savefig(os.path.join(full_path, sim_id + "_" + exp_type + "_rates_mean_colour_I.png"), bbox_inches = 'tight')

	# === Plot 2 - Colour grid of standard deviations

	# Merge all stds of E & I into one list in order to find max --> color bar legend limits
	E_I_all_stds_concatenated = np.concatenate((array_rates_std_e, array_rates_std_i), axis = 0)
	E_I_all_stds_merged = list(itertools.chain.from_iterable(E_I_all_stds_concatenated)) 

	# Input_to_E population: standard deviation
	plt.close("all")

	fig  =  plt.figure(figsize = (21, 21))
	ax1 = fig.add_subplot(1, 1, 1)
	plt.pcolor(array_rates_std_input_e.transpose(), cmap = colourmap_inputs, vmin = 0, vmax = np.nanmax(array_rates_std_input_e))

	ax1.set_xticks(np.arange(0.5, len(var_range[0]), 1))
	ax1.set_xticklabels(var_range[0], rotation = 40)
	ax1.set_yticks(np.arange(0.5, len(var_range[1]), 1))
	ax1.set_yticklabels(var_range[1])

	plt.xlabel(xlabel, size = s2, labelpad = 7)
	plt.ylabel(ylabel[0], size = s2, labelpad = 15)
	plt.xticks(size = s1)
	plt.yticks(size = s1)

	ax1.set_xlim(xmin = 0, xmax = len(var_range[0]))
	ax1.set_ylim(ymin = 0, ymax = len(var_range[1]))

	cb  =  colorbar()
	cb.set_label('Standard deviation (Hz)', size = s1, labelpad = 35)
	cb.ax.tick_params(width = lwdth, labelsize = s1, pad = 10)

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 10)

	plt.savefig(os.path.join(full_path, sim_id + "_" + exp_type + "_rates_std_colour_Input_to_E.png"), bbox_inches = 'tight')

	# Input_to_I population: standard deviation
	plt.close("all")

	fig  =  plt.figure(figsize = (21, 21))
	ax1 = fig.add_subplot(1, 1, 1)
	plt.pcolor(array_rates_std_input_i.transpose(), cmap = colourmap_inputs, vmin = 0, vmax = np.nanmax(array_rates_std_input_i))

	ax1.set_xticks(np.arange(0.5, len(var_range[0]), 1))
	ax1.set_xticklabels(var_range[0], rotation = 40)
	ax1.set_yticks(np.arange(0.5, len(var_range[1]), 1))
	ax1.set_yticklabels(var_range[1])

	plt.xlabel(xlabel, size = s2, labelpad = 7)
	plt.ylabel(ylabel[1], size = s2, labelpad = 15)
	plt.xticks(size = s1)
	plt.yticks(size = s1)

	ax1.set_xlim(xmin = 0, xmax = len(var_range[0]))
	ax1.set_ylim(ymin = 0, ymax = len(var_range[1]))

	cb  =  colorbar() 
	cb.set_label('Standard deviation (Hz)', size = s1, labelpad = 35)
	cb.ax.tick_params(width = lwdth, labelsize = s1, pad = 10)

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 10)

	plt.savefig(os.path.join(full_path, sim_id + "_" + exp_type + "_rates_std_colour_Input_to_I.png"), bbox_inches = 'tight')

	# Excitatory population: standard deviation
	plt.close("all")

	fig  =  plt.figure(figsize = (21, 21))
	ax1 = fig.add_subplot(1, 1, 1)
	plt.pcolor(array_rates_std_e.transpose(), cmap = colourmap_e, vmin = 0, vmax = np.nanmax(E_I_all_stds_merged))

	ax1.set_xticks(np.arange(0.5, len(var_range[0]), 1))
	ax1.set_xticklabels(var_range[0], rotation = 40)
	ax1.set_yticks(np.arange(0.5, len(var_range[1]), 1))
	ax1.set_yticklabels(var_range[1])

	plt.xlabel(xlabel, size = s2, labelpad = 7)
	plt.ylabel(ylabel[0], size = s2, labelpad = 15)
	plt.xticks(size = s1)
	plt.yticks(size = s1)

	ax1.set_xlim(xmin = 0, xmax = len(var_range[0]))
	ax1.set_ylim(ymin = 0, ymax = len(var_range[1]))

	cb  =  colorbar() 
	cb.set_label('Standard deviation (Hz)', size = s1, labelpad = 35)
	cb.ax.tick_params(width = lwdth, labelsize = s1, pad = 10)

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 10)

	plt.savefig(os.path.join(full_path, sim_id + "_" + exp_type + "_rates_std_colour_E.png"), bbox_inches = 'tight')

	# Inhibitory population: standard deviation
	plt.close("all")

	fig  =  plt.figure(figsize = (21, 21))
	ax1 = fig.add_subplot(1, 1, 1)
	plt.pcolor(array_rates_std_i.transpose(), cmap = colourmap_i, vmin = 0, vmax = np.nanmax(E_I_all_stds_merged))

	ax1.set_xticks(np.arange(0.5, len(var_range[0]), 1))
	ax1.set_xticklabels(var_range[0], rotation = 40)
	ax1.set_yticks(np.arange(0.5, len(var_range[1]), 1))
	ax1.set_yticklabels(var_range[1])

	plt.xlabel(xlabel, size = s2, labelpad = 7)
	plt.ylabel(ylabel[1], size = s2, labelpad = 15)
	plt.xticks(size = s1)
	plt.yticks(size = s1)

	ax1.set_xlim(xmin = 0,  xmax = len(var_range[0]))
	ax1.set_ylim(ymin = 0,  ymax = len(var_range[1]))

	cb  =  colorbar() 
	cb.set_label('Standard deviation (Hz)', size = s1, labelpad = 35)
	cb.ax.tick_params(width = lwdth, labelsize = s1, pad = 10)

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 10)

	plt.savefig(os.path.join(full_path, sim_id + "_" + exp_type + "_rates_std_colour_I.png"), bbox_inches = 'tight')
