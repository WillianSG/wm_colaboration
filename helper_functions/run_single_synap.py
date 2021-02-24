# -*- coding: utf-8 -*-
"""
@author: wgirao

input:

output:

Comments:
"""
from brian2 import *
from load_neurons import *
from numpy import mean
# from poisson_spiking_gen_noseed import *

def run_single_synap(
	pre_rate, 
	post_rate, 
	t_run, 
	dt_resolution, 
	plasticity_rule, 
	neuron_type, 
	noise, 
	bistability, 
	N_Pre, 
	N_Post, 
	tau_xpre, 
	tau_xpost, 
	xpre_jump, 
	xpost_jump, 
	rho_neg, 
	rho_neg2, 
	rho_init, 
	tau_rho, 
	thr_post, 
	thr_pre, 
	thr_b_rho, 
	rho_min, 
	rho_max, 
	alpha, 
	beta, 
	xpre_factor, 
	w_max, 
	model_E_E, 
	pre_E_E, 
	post_E_E, 
	int_meth_syn = 'euler',
	w_init = 0.5):

	Pre = PoissonGroup(
		N = N_Pre,
		rates = pre_rate*Hz)

	Post = PoissonGroup(
		N = N_Post,
		rates = post_rate*Hz)

	# Spike time arrays
	# pre_spikes_t, post_spikes_t = poisson_spiking_gen_noseed(
	# 	rate_pre = pre_rate, 
	# 	rate_post = post_rate, 
	# 	t_run = t_run, 
	# 	dt = dt_resolution, 
	# 	noise = noise)

	# # Brian2's NeuronGroup
	# Pre, Post = load_neurons(
	# 	N_Pre, N_Post, neuron_type,
	# 	spikes_t_Pre = pre_spikes_t,
	# 	spikes_t_Post = post_spikes_t,
	# 	pre_rate = pre_rate,
	# 	post_rate =  post_rate)

	Pre_Post = Synapses(
		source = Pre,
		target = Post,
		model = model_E_E,
		on_pre = pre_E_E,
		on_post = post_E_E,
		method = int_meth_syn,
		name = 'Pre_Post')

	Pre_Post.connect(j = 'i') # each in source connected to one in target

	# - Initialization of synaptic variables
	Pre_Post.rho = rho_init
	num_Pre_Post_synaspes = len(Pre_Post.i)

	# - Monitors
	synaptic_mon = StateMonitor(
		Pre_Post, 
		['xpre', 'xpost', 'rho', 'w'], 
		record = True)

	Pre_mon = SpikeMonitor(
		source = Pre, 
		record = True, 
		name = 'Pre_mon')

	Post_mon = SpikeMonitor( 
		source = Post,
		record = True,
		name = 'Post_mon')

	# Network
	defaultclock.dt = dt_resolution*second

	net = Network(
		Pre, 
		Post, 
		Pre_Post, 
		Pre_mon, 
		Post_mon, 
		synaptic_mon, 
		name = 'net')

	net.run(t_run*second) # simulating (running network)

	if w_init == 0.0: # assessing LTP probability
		if Pre_Post.rho[0] > 0.5:
			return 1
		else:
			return 0
	elif w_init == 1.0: # assessing LTD probability
		if Pre_Post.rho[0] < 0.5:
			return 1
		else:
			return 0
	else:
		return Pre_Post.rho[0] # last value of 'rho' at the end of the ex.