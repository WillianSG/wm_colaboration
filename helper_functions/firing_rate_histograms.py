# -*- coding: utf-8 -*-
"""
@author: slehfeldt

Inputs:
- tpoints: array with time points of spikes
- inds: array with neuron indices belonging to tpoints
- bin_width: desired bin width of time_resolved histogram 
- for tpoints without units: bin_width in seconds and also without unit
- for tpoints with units: bin_width with unit
- N_pop: population size
- flag_hist: string defining which type of histogram should 
- be generated. Options are 'time_resolved' and'neuron_resolved'

Outputs:
- for 'time_resolved' histograms: 
t_hist_count 
t_hist_edgs 
t_hist_fr
- for 'neuron_resolved' histograms: 
n_hist_count 
n_hist_edgs 
n_hist_fr

Comments:
- Returns a spike count and firing rate histogram. Histograms are either 'time_resolved' (population firing rate over time) or 'neuron-resolved' (firing rate of single neurons). Both histograms are calculated over the time where the population was active.
"""

def firing_rate_histograms(tpoints, inds, bin_width, N_pop, flag_hist):
	import numpy as np
	import sys

	# Calculating time where population was active

	# if tpoints contains spikes
	if len(tpoints) > 0:
		t_start = tpoints[0]
		t_end = tpoints[-1]
		t_active = t_end - t_start
	else:
		t_active = 0

	# === Time-resolved histogram

	if flag_hist == 'time_resolved':
		if len(tpoints) > 0 :      
			# a) Calculate number of bins with desired bin_width
			num_bins = t_active / bin_width
			num_bins_rounded = int(round(num_bins))
		else:
			# Set arbitrary number of bins
			num_bins_rounded = 1

		if num_bins_rounded < 1:
			num_bins_rounded = 1

		# b) Time-resolved histogram: count
		t_hist_count, t_hist_edgs = np.histogram(a = tpoints,
			bins = num_bins_rounded) 

		# c) Calculate individual width of each bin 
		t_hist_bin_widths = []
		for i in np.arange(0, num_bins_rounded, 1):
			bin_width_temp = t_hist_edgs[i + 1] - t_hist_edgs[i]
			t_hist_bin_widths.append(bin_width_temp)

		# d) Time resolved hisogram: firing rate
		t_hist_fr = t_hist_count / t_hist_bin_widths / N_pop

		return t_hist_count, t_hist_edgs, t_hist_bin_widths, t_hist_fr

	# === Neuron-resolved histogram

	if flag_hist == 'neuron_resolved':
		# a) Neuron-resolved histogram: count
		n_hist_count, n_hist_edgs = np.histogram(a = inds, 
			bins = N_pop)
		                            
		# b) Neuron-resolved histogram: firing rate
		n_hist_fr = n_hist_count / t_active

		return n_hist_count, n_hist_edgs, n_hist_fr

	sys.exit('ERROR - firing_rate_histograms.py exit with no return')