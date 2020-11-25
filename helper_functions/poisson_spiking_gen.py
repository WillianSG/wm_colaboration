# -*- coding: utf-8 -*-
"""
@author: wgirao
@author: o.j.richter@rug.nl
@based-on: asonntag
"""

# Creates the arrays containing the spiking times for both pre- and postsynaptic neurons.
"""
input:
 - rate_pre: pre synaptic rate in Hz
 - rate_post: post synaptic rate in Hz
 - t_run: time steps to run
 - dt: time step in sec
 - noise: noise multipier for low_freq ( max/min of uniform distribution is 1/high_feq*noise)
 - seed: default 0, the seed for the random generator, for reproducibility numpy version must match.
 - correlation: random (default): uniform balanced correlation - zero centered, 
                positive: only the positive shift of the random uniform distribution, 
                negative: only the negative shift of the random uniform distribution.

output:
- pre_spikes_t (numpy array): pres- spikes times
- post_spikes_t: post- spikes times

Comments:
"""
import numpy as np
from numpy.random import Generator, PCG64, SeedSequence
import logging
from shift_close_spikes import *

def poisson_spiking_gen(rate_pre, rate_post, t_run, dt, noise, seed=0, correlation="random"):
	# Initializing decision variables
	rate_low = -1
	rate_high = -1

	# Initialise random gen
	# from seed to have reproducible results, numpy version must match!
	seedsequ = SeedSequence(seed)
	logging.info('the following random seed is used for spike generation: {}'.format(seedsequ.entropy))
	randomgen = Generator(PCG64(seedsequ))

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
		vt_highrate = randomgen.random(np.size(times))

		# Indexes where spike happened
		spikes_highrate = (spikes_per_s * dt) > vt_highrate

		# Array containing the spike times of the neuron spiking at the higher rate.
		"""
		'np.argwhere(spikes_highrate)' is the index of the array 'vt_highrate' where the random number was smaller than '(spikes_per_s*dt)' (chance of spike in each time step);
		"""
		highrate_spikes_t = np.argwhere(spikes_highrate) * dt

		if rate_low == 0:
			lowrate_spikes_t = np.array([])
		elif rate_low > 0:
			spike_prob = rate_low / rate_high
			rand_num = randomgen.random(len(highrate_spikes_t))
			# Array containing the spike times of the neuron spiking at the lower rate.
			"""
			Only gets the values 'highrate_spikes_t[x]', where x is the index of 'rand_num' for which the value is smaller than 'spike_prob'.
			"""
			lowrate_spikes_t = highrate_spikes_t[rand_num < spike_prob]

			# @TODO we should think about not making the distribution adaptive, but fixed to 1-3* STDP time constant?
			# be aware of impact on correlation
			avg_interval = 1.0 / spikes_per_s
			# 'len(lowrate_spikes_t)'s floats between ['-avg_interval*noise','avg_interval*noise')
			shifts = randomgen.uniform(-avg_interval * noise, avg_interval * noise, len(lowrate_spikes_t))

			if correlation == "random":
				pass
			elif correlation == "positive" or correlation == "negative":
				shifts = np.abs(shifts)
			else:
				raise KeyError("correlation can only be: random (default), positive, negative")

			# 'lowrate_spikes_t' has now some of the same spike times as 'highrate_spikes_t' but with a shifted value (for more or for less)
			lowrate_spikes_t = lowrate_spikes_t.flatten() + shifts

			# Removing negative values
			lowrate_spikes_t = lowrate_spikes_t[lowrate_spikes_t > 0]

		# Removing spikes that are too close - QUESTION (why only with lowrate?) ANSWER: because of the added noise only to low rate
		# I would not even remove close spikes because otherwise you will just have 2 excecuted shortly after one another
		lowrate_spikes_t = shift_close_spikes(np.sort(lowrate_spikes_t), dt, t_run)

		# remove spikes that have negative time steps (not allowed)
		lowrate_spikes_t = lowrate_spikes_t[lowrate_spikes_t > 0]

		# Updating return values
		if rate_pre >= rate_post:
			pre_spikes_t = np.sort(highrate_spikes_t)
			post_spikes_t = np.sort(lowrate_spikes_t)
		elif rate_pre <= rate_post:
			post_spikes_t = np.sort(highrate_spikes_t)
			pre_spikes_t = np.sort(lowrate_spikes_t)

	# quick-fix bug for negative correlation
	if correlation == "negative":
		temp = post_spikes_t
		post_spikes_t = pre_spikes_t
		pre_spikes_t = temp

	return pre_spikes_t, post_spikes_t
