# -*- coding: utf-8 -*-
"""
@author: slehfeldt

Input:

Output:
"Workflow: 1) From n_inds and s_tpoints: match neuron indices with spike times
		2) Calculate inter spike intervals"

Comments:
- Calculates the inter spike intervals from a population. 
- The function matches active neuron indices to their spike times and calculates the inter spike intervals for every spiketrain per neuron. 
- Neuron with only one spike are skipped since there is no interval to calculate from one spike.
"""
import numpy as np

def inter_spike_intervals(n,inds,tpoints):
	isi_list = []
	isi_temp = []

	# 1) From n_inds and s_tpoints: match neuron indices with spike times
	for i in np.arange(0, n): # Loop through every neuron
		# Identify index positions where neuron was active
		temp_inds = numpy.where(inds == i)

		# Only if more than one spike
		if len(temp_inds[0]) > 1:
			temp_tpoints = tpoints[temp_inds[0]] # Get spikes time points
				# 2) Calculate isi of neuron
				# loop through spike times
				for j in np.arange(0, len(temp_tpoints)-1):
					isi_temp = temp_tpoints[j+1]-temp_tpoints[j] # calc. isi
					isi_list.append(isi_temp) # append interval to list

	return isi_list