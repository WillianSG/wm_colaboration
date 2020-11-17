# -*- coding: utf-8 -*-
"""
@author: wgirao
@based-on: asonntag
"""

# Creates the arrays containing the spiking times for both pre- and postsynaptic neurons.
"""
input:

output:
- pre_spikes_t (numpy array): pres- spikes times
- post_spikes_t: post- spikes times

Comments:
"""
import numpy as np
from brian2 import uniform, rand, size
from remove_close_spikes import *

def poisson_spiking_gen(rate_pre, rate_post, t_run, dt, noise):
	# Initializing decision variables
	rate_low = -1
	rate_high = -1
    
    # Arrays with spike times
	pre_spikes_t = np.array([])
	post_spikes_t = np.array([])

	if rate_pre > 0 or rate_post > 0:
		# Deciding highest/lowest frequencies
		if rate_pre >= rate_post:
			rate_high = rate_pre
			rate_low = rate_post
		else:
			rate_high = rate_post
			rate_low = rate_pre

		# Average number os spikes per second
		spikes_per_s = rate_high

		# Array with each time step of simulation
		times = np.arange(0, t_run, dt)

		# random number (in [0,1)) for each time step (membrane threshold for spike gen?) 
		vt_highrate = rand(size(times))

		# Indexes where spike happened
		spikes_highrate = (spikes_per_s*dt) > vt_highrate

		# Array containing the spike times of the neuron spiking at the higher rate.
		"""
		'np.argwhere(spikes_highrate)' is the index of the array 'vt_highrate' where the random number was smaller than '(spikes_per_s*dt)' (chance of spike in each time step);
		"""
		highrate_spikes_t = np.argwhere(spikes_highrate)*dt

		if rate_low == 0:
			lowrate_spikes_t = np.array([])
		elif rate_low > 0:
			spike_prob = rate_low/rate_high
			rand_num = rand(len(highrate_spikes_t))
			# Array containing the spike times of the neuron spiking at the lower rate.
			"""
			Only gets the values 'highrate_spikes_t[x]', where x is the index of 'rand_num' for which the value is smaller than 'spike_prob'.
			"""
			lowrate_spikes_t = highrate_spikes_t[rand_num < spike_prob]

			# ATENTION - experiment with negative shift btween spike timings should be from here on.

			avg_interval = 1.0/spikes_per_s

			# 'len(lowrate_spikes_t)'s floats between ['-avg_interval*noise','avg_interval*noise')
			shifts = uniform(-avg_interval*noise, avg_interval*noise, len(lowrate_spikes_t))

			# 'lowrate_spikes_t' has now some of the same spike times as 'highrate_spikes_t' but with a shifted value (for more or for less) 
			lowrate_spikes_t = lowrate_spikes_t.flatten()+shifts

			# Removing negative values
			lowrate_spikes_t = lowrate_spikes_t[lowrate_spikes_t > 0]

			# Removing spikes that are too close - QUESTION (why only with lowrate?)
			lowrate_spikes_t = remove_close_spikes(np.sort(lowrate_spikes_t), dt, t_run)

			# ALTERED (S) - I guess this should be moved to be exec. before removing close spikes | moved to line 79
			# lowrate_spikes_t = lowrate_spikes_t[lowrate_spikes_t > 0]

		# Updating return values
		if rate_pre >= rate_post:
			pre_spikes_t = np.sort(highrate_spikes_t)
			post_spikes_t = np.sort(lowrate_spikes_t)
		elif rate_pre <= rate_post:
			post_spikes_t = np.sort(highrate_spikes_t)
			pre_spikes_t = np.sort(lowrate_spikes_t)

	return pre_spikes_t, post_spikes_t

