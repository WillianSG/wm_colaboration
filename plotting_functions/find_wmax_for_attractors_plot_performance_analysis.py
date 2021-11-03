# -*- coding: utf-8 -*-
"""
@author: Lehfeldt with some adaptations by asonntag and wgirao 

Input:

Output:
- counts:
- mean_attractor_frequencies:
- std_attractor_frequencies:
- attractor_frequencies_classified:

Comments:
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from brian2 import mV, mean, std

def find_wmax_for_attractors_plot_performance_analysis(path_sim, sim_id, exp_type, num_networks, sim_folders_list, simulation_flags_wmax, wmax_range, delay_activities, attractor_frequencies):

	lwdth = 2
	s1 = 30
	s2 = 60
	mpl.rcParams['axes.linewidth'] = lwdth

	# 1 - Counts of sucessful delay activity

	count_da = np.zeros(len(wmax_range))
	count_fading_da = np.zeros(len(wmax_range))
	count_no_da = np.zeros(len(wmax_range))
	dataset_temp = 0

	for i in np.arange(0, len(delay_activities), 1):
		dataset_temp = delay_activities[sim_folders_list[i]]
		flag_wmax_pos = simulation_flags_wmax[i][0]

		# No delay activity
		if dataset_temp.count(True) == 0:
			count_no_da[flag_wmax_pos] += 1

		# Fading delay activity    
		if dataset_temp.count(True) == 1:
			count_fading_da[flag_wmax_pos] += 1

		# Delay activity
		if dataset_temp.count(True) == 2:
			count_da[flag_wmax_pos] += 1

	all_counts_concatenated = np.concatenate((count_da, count_fading_da, count_no_da), axis = 0)

	counts = {}
	counts['No delay activity'] = count_no_da
	counts['Fading delay activity'] = count_fading_da
	counts['Delay activity'] = count_da
	counts['Wmax range (mV)'] = wmax_range

	# 2.1 Plotting counts as bar charts

	os.chdir(path_sim)
	plt.close('all')

	N = len(sim_folders_list[0:-1])/len(wmax_range)
	bar_width = 0.065 # 0.2 settings for find_wmax plots
	x = wmax_range - 2*bar_width/2

	fig = plt.figure(figsize = (25, 17.5))

	ax0 = fig.add_subplot(1, 1, 1)

	p2 = ax0.bar(x, count_fading_da, bar_width, color = 'lightgrey', edgecolor = 'black', linewidth = lwdth)

	p3 = ax0.bar(x + bar_width, count_da, bar_width, color = 'dimgray', edgecolor = 'black',  linewidth = lwdth)

	ax0.legend((p2[0],p3[0]), ('Fading DA', 'DA'), prop = {'size' : 30}, bbox_to_anchor = (1, 1), ncol = 3)

	plt.xlabel('Maximal weight $w_{max}$ (mV)', size = s1, labelpad = 15)
	plt.ylabel('Count', size = s1, labelpad = 15)
	plt.yticks(size = s1)
	plt.xticks(size = s1)

	plt.ylim(-max(all_counts_concatenated)/10, max(all_counts_concatenated)*1.1) # settings for find_wmax plot

	plt.tick_params(axis = 'both',which = 'major',width = lwdth,length = 5,pad = 10)

	ax0.set_xticks(wmax_range)

	plt.savefig(os.path.join(path_sim,sim_id + '_learning_performance.png'), bbox_inches = 'tight')

	# 2.2 Attractor frequencies

	# Containers for frequency collection in dependence of classification result  
	attractor_frequencies_no_da = {}
	attractor_frequencies_fading_da = {}
	attractor_frequencies_da = {}

	# Create collectors for frequencies
	for i in np.arange(0, len(wmax_range), 1):
		attractor_frequencies_no_da[i] = []
		attractor_frequencies_fading_da[i] = []
		attractor_frequencies_da[i] = []

	# Loop through simulations
	for j in np.arange(0, len(attractor_frequencies), 1):
		dataset_temp = delay_activities[sim_folders_list[i]]
		freq_temp = attractor_frequencies[sim_folders_list[j]]
		flag_wmax_pos = simulation_flags_wmax[j][0] 

		# No delay activity
		if dataset_temp.count(True) == 0:
			attractor_frequencies_no_da[flag_wmax_pos].append(freq_temp)

		# Fading delay activity    
		if dataset_temp.count(True) == 1: 
			attractor_frequencies_fading_da[flag_wmax_pos].append(freq_temp)

		# Delay activity
		if dataset_temp.count(True) == 2:
			attractor_frequencies_da[flag_wmax_pos].append(freq_temp)

	mean_freqs_no_da = []
	mean_freqs_fading_da = [] 
	mean_freqs_da = [] 
	std_freqs_no_da = []
	std_freqs_fading_da = []
	std_freqs_da = []

	for k in np.arange(0, len(attractor_frequencies_no_da), 1):
		mean_freqs_no_da.append(np.mean(attractor_frequencies_no_da[k]))
		std_freqs_no_da.append(std(attractor_frequencies_no_da[k]))

		mean_freqs_fading_da.append(np.mean(attractor_frequencies_fading_da[k]))
		std_freqs_fading_da.append(np.std(attractor_frequencies_fading_da[k]))
		
		mean_freqs_da.append(np.mean(attractor_frequencies_da[k]))
		std_freqs_da.append(np.std(attractor_frequencies_da[k]))

	os.chdir(path_sim)

	fig = plt.figure(figsize = (25, 17.5))
	ax0 = fig.add_subplot(2, 1, 1)

	plt.errorbar(wmax_range, mean_freqs_no_da, yerr = std_freqs_no_da, color = 'black')
	plt.errorbar(wmax_range, mean_freqs_fading_da, yerr = std_freqs_fading_da, color = 'blue')
	plt.errorbar(wmax_range, mean_freqs_da, yerr = std_freqs_da, color = 'green')

	plt.xlabel('Maximal weight $w_{max}$ (mV)', size = s1, labelpad = 15)  
	plt.ylabel('Mean frequency +/- s.d. (Hz)', size = s1, labelpad = 15)

	plt.yticks(size = s1)
	plt.xticks(size = s1)

	plt.xlim(min(wmax_range)-0.5, max(wmax_range)+0.5)

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 5, pad = 10)

	plt.savefig(os.path.join(path_sim, sim_id + '_learning_performance_freqs.png'), bbox_inches = 'tight')

	attractor_frequencies_classified = [attractor_frequencies_no_da,attractor_frequencies_fading_da, attractor_frequencies_da]

	mean_attractor_frequencies = [mean_freqs_no_da, mean_freqs_fading_da, mean_freqs_da]

	std_attractor_frequencies = [std_freqs_no_da, std_freqs_fading_da,std_freqs_da]

	return counts, mean_attractor_frequencies, std_attractor_frequencies,attractor_frequencies_classified





