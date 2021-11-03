# -*- coding: utf-8 -*-
"""
@author: slehfeldt

Input:
- cwd: string holding the path of the current working directory
- full_path: string holding path to simulation folder
- sim_id: string holding the simulation identifier ('YYYYMMDD_hh_mm_ss')
- N: list with population sizes
- var_range: list with variable range of experiment
- var_steps: list with step size of each running variable of experiment
- dicts_means: dictionary with mean firing rate of each loop iteration fo experiment
- dicts_stds: dictionary with standard deviations fo firing rates of each loop iteration
- exp_type: string describing the experiment  
- simulation_flag: string describing which population (E_only, I_only or E_and_I) should be plotted

Output: 

Comments:
- Plots the input to output curve (activation frequency and resulting population frequency) over the defined frequency range and for different values of the second running variable of the experiment.
"""
import os
from brian2 import mV, Hz, khertz
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def netdyn_plot_input_output_curve(full_path, sim_id, N, var_range, var_steps, dicts_means, dicts_stds, exp_type, simulation_flag):
	lwdth = 2
	s1 = 30
	s2 = 75
	mpl.rcParams['axes.linewidth'] = lwdth

	# Convert dictionaries to lists
	for var_inner in range(0, len(var_range[1])):
		list_mean_input_e = np.zeros(len(var_range[0]))
		list_std_input_e = np.zeros(len(var_range[0]))
		list_mean_input_i = np.zeros(len(var_range[0]))
		list_std_input_i = np.zeros(len(var_range[0]))
		list_mean_e = np.zeros(len(var_range[0]))
		list_std_e = np.zeros(len(var_range[0]))
		list_mean_i = np.zeros(len(var_range[0]))
		list_std_i = np.zeros(len(var_range[0]))

		for var_outer in range(0, len(var_range[0])): 
			list_mean_input_e[var_outer] = dicts_means[0][var_outer, var_inner]
			list_std_input_e[var_outer] = dicts_stds[0][var_outer, var_inner]
			list_mean_input_i[var_outer] = dicts_means[1][var_outer, var_inner]
			list_std_input_i[var_outer] = dicts_stds[1][var_outer, var_inner]
			list_mean_e[var_outer] = dicts_means[2][var_outer, var_inner]
			list_std_e[var_outer] = dicts_stds[2][var_outer, var_inner]
			list_mean_i[var_outer] = dicts_means[3][var_outer, var_inner]
			list_std_i[var_outer] = dicts_stds[3][var_outer, var_inner]

		if simulation_flag == 'E_only':
			fig = plt.figure(figsize = (25,  8.75))
			ax1 = fig.add_subplot(1, 2, 2)

			# Excitatory population: mean +/- std
			plt.plot(var_range[0]/khertz, list_mean_e, 'o-', color = 'mediumblue', ms = 10, linewidth = lwdth)
			plt.fill_between(var_range[0]/khertz, list_mean_e-list_std_e, list_mean_e+list_std_e, facecolor = 'mediumblue', alpha = 0.5, edgecolor = 'none')

			plt.xlabel('Activation frequency (kHz)', size = s1, labelpad = 20)
			plt.ylabel('Mean $FR_{E}$ (Hz)', size = s1, labelpad = 30)
			plt.xticks(size = s1-10)
			plt.yticks(size = s1)

			ax1.set_xlim(xmin = min(var_range[0])/khertz-0.3, xmax = max(var_range[0])/khertz+0.3)
			ax1.set_ylim(-5, 110)

			plt.tick_params(which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 15)

			# Input population
			ax2 = fig.add_subplot(1, 2, 1)

			# Input populations: mean +/- std
			plt.plot(var_range[0]/khertz, list_mean_input_e/khertz, 'o-', color = 'grey', ms = 10, linewidth = lwdth)

			plt.fill_between(var_range[0]/khertz, list_mean_input_e/khertz-list_std_input_e/khertz, list_mean_input_e/khertz+list_std_input_e/khertz, facecolor = 'grey', alpha = 0.5, edgecolor = 'none')

			plt.xlabel('Activation frequency (kHz)', size = s1, labelpad = 20)
			plt.ylabel('Mean $FR_{Input E}$ (kHz)', size = s1, labelpad = 30)

			plt.xticks(size = s1-10)
			plt.yticks(size = s1)

			plt.tick_params(which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 15)

			ax2.set_xlim(xmin = min(var_range[0])/khertz-0.3, xmax = max(var_range[0])/khertz+0.3)
			ax2.set_ylim(ymin = min(var_range[0])/khertz-0.3, ymax = max(var_range[0])/khertz+0.3)

			plt.subplots_adjust(wspace = 0.3)

			plt.savefig(os.path.join(full_path, sim_id + '_' + exp_type + '_IO_curve_weight_' + format(var_range[1][var_inner]/mV, '09.3f') + '_mV.png'), bbox_inches = 'tight', dpi = 200)

		if simulation_flag == 'I_only':
			fig  =  plt.figure(figsize = (25, 8.75))
			ax1  =  fig.add_subplot(1, 2, 2)

			# Excitatory population: mean +/- std
			plt.plot(var_range[0]/khertz, list_mean_i, 'o-', color = 'green', ms = 10, linewidth = lwdth)
			plt.fill_between(var_range[0]/khertz, list_mean_i-list_std_i, list_mean_i+list_std_i, facecolor = 'green', alpha = 0.5, edgecolor = 'none')

			plt.xlabel('Activation frequency (kHz)', size = s1, labelpad = 20)
			plt.ylabel('Mean $FR_{I}$ (Hz)', size = s1, labelpad = 30)

			plt.xticks(size = s1-10)
			plt.yticks(size = s1)

			ax1.set_xlim(xmin = min(var_range[0])/khertz-0.3, xmax = max(var_range[0])/khertz+0.3)
			ax1.set_ylim(-5, 110)

			plt.tick_params(which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 15)

			# Input population
			ax2  =  fig.add_subplot(1, 2, 1)

			# Input populations: mean +/- std
			plt.plot(var_range[0]/khertz, list_mean_input_i/khertz, 'o-', color = 'grey', ms = 10, linewidth = lwdth)
			plt.fill_between(var_range[0]/khertz, list_mean_input_i/khertz-list_std_input_i/khertz, list_mean_input_i/khertz+list_std_input_i/khertz, facecolor = 'grey', alpha = 0.5, edgecolor = 'none')

			plt.xlabel('Activation frequency (kHz)', size = s1, labelpad = 20)
			plt.ylabel('Mean $FR_{Input I}$ (kHz)', size = s1, labelpad = 30)

			plt.xticks(size = s1-10)
			plt.yticks(size = s1)

			plt.tick_params(which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 15)

			ax2.set_xlim(xmin = min(var_range[0])/khertz-0.3, xmax = max(var_range[0])/khertz+0.3)
			ax2.set_ylim(ymin = min(var_range[0])/khertz-0.3, ymax = max(var_range[0])/khertz+0.3)

			plt.subplots_adjust(wspace = 0.3)

			plt.savefig(os.path.join(full_path,  sim_id + '_' + exp_type + '_IO_curve_weight_' + format(var_range[1][var_inner]/mV, '09.3f') + '_mV.png'), bbox_inches = 'tight', dpi = 200)

		if simulation_flag == 'weight_wee':
			fig  =  plt.figure(figsize = (17.5, 25))
			ax1  =  fig.add_subplot(2, 1, 1)

			# Excitatory population: mean +/- std
			plt.plot(var_range[0]/mV, list_mean_e, 'o-', color = 'mediumblue', ms = 20, linewidth = lwdth, label = '$E$')
			plt.plot(var_range[0]/mV, list_mean_i, 'o-', color = 'green', ms = 20, linewidth = lwdth, label = '$I$')

			plt.fill_between(var_range[0]/mV, list_mean_e-list_std_e, list_mean_e+list_std_e, facecolor = 'mediumblue', alpha = 0.5, edgecolor = 'none')
			plt.fill_between(var_range[0]/mV, list_mean_i-list_std_i, list_mean_i+list_std_i, facecolor = 'green', alpha = 0.5, edgecolor = 'none')

			plt.legend(loc = 'upper left', prop = {'size':35})

			plt.xlabel('$w_{EE}$ (mV)', size = s1, labelpad = 20)
			plt.ylabel('Mean $FR$ (Hz)', size = s1, labelpad = 30)

			plt.xticks(size = s1-10)
			plt.yticks(size = s1)

			ax1.set_xlim(xmin = min(var_range[0])/mV, xmax = max(var_range[0])/mV)
			ax1.set_ylim(0, max(list_mean_e[:])*1.1)

			plt.tick_params(which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 15)

			plt.savefig(os.path.join(full_path, sim_id + '_' + exp_type + '_IO_curve_wee.png'), bbox_inches = 'tight',  dpi = 200)


		if simulation_flag == 'E_and_I':
			fig  =  plt.figure(figsize = (25, 25))

			plt.gcf().text(0.04, 0.925, 'W_input_e: %s mV' %(var_range[1][var_inner]/mV), fontsize = s1, fontweight = 'bold')

			# Excitatory and inhibitory population
			ax1  =  fig.add_subplot(2, 1, 1)
			plt.plot(var_range[0], var_range[0]/100, linewidth = lwdth, color = 'grey', label = 'Input  =  Output')

			# Excitatory population: mean +/- std
			plt.plot(var_range[0], list_mean_e, 'o-', color = 'mediumblue', ms = 20, linewidth = lwdth, label = 'Excitatory neurons')
			plt.fill_between(var_range[0], list_mean_e-list_std_e, list_mean_e+list_std_e, facecolor = 'mediumblue', alpha = 0.5, edgecolor = 'none')

			# Inhibitory population: mean +/- std
			plt.plot(var_range[0], list_mean_i, 'o-', color = 'green', ms = 20, linewidth = lwdth, label = 'Inhibitory neurons')
			plt.fill_between(var_range[0], list_mean_i-list_std_i, list_mean_i+list_std_i, facecolor = 'green', alpha = 0.5, edgecolor = 'none')

			plt.legend(loc = 'upper left', prop = {'size':s1})

			plt.xlabel('Activation frequency (Hz)', size = s1, labelpad = 20)
			plt.ylabel('Population frequency (Hz)', size = s1, labelpad = 30)

			plt.xticks(size = s1-10)
			plt.yticks(size = s1)

			ax1.set_xlim(xmin = min(var_range[0])/Hz, xmax = max(var_range[0])/Hz)
			ax1.set_ylim(ymin = -50, ymax = 150)

			plt.tick_params(which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 15)

			# Input populations
			ax2  =  fig.add_subplot(2, 1, 2)
			plt.plot(var_range[0], var_range[0], linewidth = lwdth, color = 'grey', label = 'Input  =  Output')

			# Input populations: mean +/- std
			plt.plot(var_range[0], list_mean_input_e, 'o-', color = 'grey', ms = 20, linewidth = lwdth, label = 'Input_to_E neurons')
			plt.plot(var_range[0], list_mean_input_i, 'o-', color = 'black', ms = 20, linewidth = lwdth, label = 'Input_to_I neurons')

			plt.fill_between(var_range[0], list_mean_input_e-list_std_input_e, list_mean_input_e+list_std_input_e, facecolor = 'grey', alpha = 0.5, edgecolor = 'none')
			plt.fill_between(var_range[0], list_mean_input_i-list_std_input_i, list_mean_input_i+list_std_input_i, facecolor = 'black', alpha = 0.5, edgecolor = 'none')

			plt.legend(loc = 'upper left', prop = {'size':s1})

			plt.xlabel('Activation frequency (Hz)', size = s1, labelpad = 20)
			plt.ylabel('Population activity (Hz)', size = s1, labelpad = 30)

			plt.xticks(size = s1-10)
			plt.yticks(size = s1)

			plt.tick_params(which = 'major', width = lwdth, length = 9, labelsize = s1, pad = 15)

			ax2.set_xlim(xmin = min(var_range[0])/Hz, xmax = max(var_range[0])/Hz)

			plt.savefig(os.path.join(full_path, sim_id + '_' + exp_type + '_IO_curve_weight_' + format(var_range[1][var_inner]/mV, '09.3f') + '_mV.png'), bbox_inches = 'tight', dpi = 200)
