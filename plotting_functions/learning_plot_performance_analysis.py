# -*- coding: utf-8 -*-
"""
@author: asonntag adapted from Lehfeldt

Inputs:

Outputs:
- counts
- mean_attractor_frequencies
- std_attractor_frequencies
- attractor_frequencies

Comments:
"""

def learning_plot_performance_analysis(path_sim, sim_id, exp_type, num_networks, sim_folders_list, simulation_flags_pulse_duration, pulse_durations, delay_activities, attractor_frequencies):
	import os
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	import numpy as np
	from brian2 import second, mean, std

	# General plotting settings
	lwdth = 2
	s1 = 30
	s2 = 60
	mpl.rcParams['axes.linewidth'] = lwdth

	# === 1) Counts of sucessful delay activities

	# Counters
	count_da = np.zeros(len(pulse_durations))
	count_fading_da = np.zeros(len(pulse_durations))

	count_no_da = np.zeros(len(pulse_durations))

	for i in np.arange(0, len(delay_activities), 1):
		dataset_temp = delay_activities[sim_folders_list[i]]

		flag_pulse_duration_pos = simulation_flags_pulse_duration[i][0]

		# No delay activity
		if dataset_temp.count(True) == 0:
			count_no_da[flag_pulse_duration_pos] += 1

		# Fading delay activity    
		if dataset_temp.count(True) == 1:
			count_fading_da[flag_pulse_duration_pos] += 1

		# Delay activity
		if dataset_temp.count(True) == 2:
			count_da[flag_pulse_duration_pos] += 1

	all_counts_concatenated = np.concatenate((count_da, count_fading_da,
		count_no_da), axis = 0) 

	counts = {}
	counts['No delay activity'] = count_no_da
	counts['Fading delay activity'] = count_fading_da
	counts['Delay activity'] = count_da
	counts['Stimulus durations (s)'] = pulse_durations

	# Plot: Counts as bar charts
	os.chdir(path_sim)
	plt.close('all')

	N = len(sim_folders_list[0:-1])/len(pulse_durations)
	bar_width = 0.25

	# Set position of bar on X axis
	r1 = np.arange(len(pulse_durations))
	r2 = [x + bar_width for x in r1]
	r3 = [x + bar_width for x in r2]

	fig = plt.figure(figsize = (25, 17.5))
	ax0 = fig.add_subplot(1, 1, 1)

	p1 = ax0.bar(r1, count_no_da, bar_width, color = 'white', 
		edgecolor = 'black', hatch = '', linewidth = lwdth)

	p2 = ax0.bar(r2, count_fading_da, bar_width, color = 'lightgrey', 
		edgecolor = 'black', linewidth = lwdth)

	p3 = ax0.bar(r3, count_da, bar_width, color='dimgray', edgecolor='black', linewidth=lwdth)

	ax0.legend((p1[0], p2[0], p3[0]), ('No DA', 'Fading DA', 'DA'), prop = {'size':30}, bbox_to_anchor = (1, 1), ncol = 3)

	plt.xlabel('Stimulus duration (s)', size = s1, labelpad = 15)
	plt.ylabel('Count', size = s1, labelpad = 15)

	plt.yticks(size = s1)
	plt.xticks([r + bar_width for r in range(len(pulse_durations))], [str(int(x)) for x in pulse_durations], size = s1)

	plt.ylim(-max(all_counts_concatenated)/10, 
		max(all_counts_concatenated)*1.1)

	
	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 5,
		pad = 10)

	plt.savefig(
		os.path.join(path_sim,sim_id + '_learning_performance.png'),
		bbox_inches = 'tight')

	# === 2) Attractor frequencies

	attractor_frequencies = {}
	attractor_frequencies_with = {}

	for i in np.arange(0, len(pulse_durations), 1):
		attractor_frequencies[i] = []
		attractor_frequencies_with[i] = []

	for j in np.arange(0, len(attractor_frequencies), 1):
		freq_temp = attractor_frequencies[sim_folders_list[j]]
		flag_pulse_duration_pos = simulation_flags_pulse_duration[j][0]

	attractor_frequencies[flag_pulse_duration_pos].append(freq_temp)

	mean_attractor_frequencies = []  
	std_attractor_frequencies = []

	for k in np.arange(0, len(attractor_frequencies), 1):
		mean_attractor_frequencies.append(mean(attractor_frequencies[k]))
		std_attractor_frequencies.append(std(attractor_frequencies[k]))

	os.chdir(path_sim)

	fig = plt.figure(figsize = (25, 17.5))
	ax0 = fig.add_subplot(1, 1, 1) 

	plt.errorbar(pulse_durations, mean_attractor_frequencies, yerr = std_attractor_frequencies, color = 'black')

	plt.xlabel('Stimulus duration (s)', size = s1, labelpad = 15)
	plt.ylabel('Mean frequency +/- s.d. (Hz)', size = s1, labelpad = 15)

	plt.yticks(size = s1)
	plt.xticks(size = s1)

	plt.xlim(min(pulse_durations)-0.5, max(pulse_durations)+0.5)

	plt.tick_params(axis = 'both', which = 'major', width = lwdth, length = 5,
		pad = 10)

	plt.savefig(
		os.path.join(path_sim, sim_id + '_learning_performance_freqs.png'),
		bbox_inches = 'tight')

	return counts, mean_attractor_frequencies, std_attractor_frequencies, attractor_frequencies

