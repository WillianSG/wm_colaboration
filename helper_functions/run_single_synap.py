# -*- coding: utf-8 -*-
"""
@author: wgirao
"""

# ?
"""
input:
- path_sim_id: paths for plots
- path_sim_data: paths for plots
- sim_id: simulation id
- exp_type: type of experiment
- pre_rate: required firing frequency
- post_rate: required firing frequency
- t_run(int): time of execution (seconds)
- dt_resolution(float): simulation time step
- plasticity_rule(str): 
- neuron_type(str):
- noise(float): used to calculate the shift for spike-pair generation
- bistability(bool):
- N_Pre: number of neurons in group
- N_Post: number of neurons in group
- [tau_xpre, tau_xpost, xpre_jump, xpost_jump, rho_dep, rho_dep2, rho_init,
	tau_rho, thr_post, thr_pre, thr_b_rho, rho_min, rho_max, alpha, beta, xpre_factor, w_max] <- load_rule_params()
- [model_E_E, pre_E_E, post_E_E] <- load_synapse_model()

output:
-
-
-

Comments:
"""
from brian2 import Synapses,StateMonitor, SpikeMonitor, defaultclock, Network, second
from poisson_spiking_gen_noseed import *
from load_neurons import *
from numpy import mean

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
	isi_correlation = 'random'):

	# Spike time arrays
	pre_spikes_t, post_spikes_t = poisson_spiking_gen_noseed(
		rate_pre = pre_rate, 
		rate_post = post_rate, 
		t_run = t_run, 
		dt = dt_resolution, 
		noise = noise)

	# Brian2's NeuronGroup
	Pre, Post = load_neurons(
		N_Pre, N_Post, neuron_type,
		spikes_t_Pre = pre_spikes_t,
		spikes_t_Post = post_spikes_t,
		pre_rate = pre_rate,
		post_rate =  post_rate)

	# Synapse connection
	int_meth_neur = None

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

	print('> pos: ', post_rate, 'Hz | pre: ', pre_rate, 'Hz')

	net.run(t_run*second) # simulating (running network)

	return Pre_Post.rho[0] # last value of 'rho' at the end of the ex.