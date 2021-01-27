# -*- coding: utf-8 -*-
"""
@author: slehfeldt

Inputs:
- s_tpoints: array with time points of spikes
- n_inds: array with neuron indices belonging to tpoints
- t_run: duration of simulation  
- stim_pulse_duration: duration of stimulus
- size_attractor: size of the expected attractor / stimulus size

Outputs:
- freq_temp: calculated frequency  

Comments:
- Extracts the spiking activity of the last second of stimulation and calculates the firing rate in this temporal window.
"""

def learning_check_attractor_frequency(s_tpoints, n_inds, t_run, stim_pulse_duration, size_attractor):
	import sys
	from brian2 import second

	# Import required functions
	from extract_trial_data import extract_trial_data

	delta_t = 1*second

	t_start = stim_pulse_duration - delta_t 
	t_end = stim_pulse_duration

	[inds_temp, tpoints_temp] = extract_trial_data(
		t_start = t_start, 
		t_end = t_end,
		inds = n_inds, 
		tpoints = s_tpoints)

	freq_temp = len(tpoints_temp) / delta_t / size_attractor

	return freq_temp
