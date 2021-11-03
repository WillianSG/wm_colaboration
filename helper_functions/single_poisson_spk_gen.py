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

def single_poisson_spk_gen(rate, t_run, dt, job_seed = 0):
	# Arrays with spike times
	spikes_t = np.array([])

	if rate > 0:
		# Average number os spikes per second
		spikes_per_s = rate

		# Initialise random gen
		# from seed to have reproducible results, numpy version must match!
		seedsequ = SeedSequence(job_seed)
		logging.info('the following random seed is used for spike generation: {}'.format(seedsequ.entropy))
		randomgen = Generator(PCG64(seedsequ))

		# Array with each time step of simulation
		times = np.arange(0, t_run, dt)

		# random number (in [0,1)) for each time step (membrane threshold for spike gen?)
		vt_rate = randomgen.random(np.size(times))

		# Indexes where spike happened
		spikes = (spikes_per_s * dt) > vt_rate

		# Array containing the spike times of the neuron spiking at the higher rate.
		"""
		'np.argwhere(spikes_highrate)' is the index of the array 'vt_highrate' where the random number was smaller than '(spikes_per_s*dt)' (chance of spike in each time step);
		"""
		spikes_t = np.argwhere(spikes) * dt

		spikes_t = np.sort(spikes_t)

		spikes_t = shift_close_spikes(spikes_t, dt, t_run)

	return spikes_t
