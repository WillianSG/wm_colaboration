# -*- coding: utf-8 -*-
"""
@author: wgirao
@based-on: asonntag
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
- t_run: time of execution (units?)
- dt: ?
- plasticity_rule: 
- neuron_type:
- noise: ?
- bistability (bool):
- plot_single_trial(bool): ?
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
from brian2 import *
from poisson_spiking_gen import *
from load_neurons import *

def run_frequencies(pre_rate, post_rate, t_run, dt, plasticity_rule, neuron_type, noise, bistability, plot_single_trial, N_Pre, N_Post, tau_xpre, tau_xpost, xpre_jump, xpost_jump, rho_dep, rho_dep2, rho_init, tau_rho, thr_post, thr_pre,	thr_b_rho, rho_min, rho_max, alpha, beta, xpre_factor, w_max, model_E_E,	pre_E_E, post_E_E):

	# Spike time arrays
	pre_spikes_t, post_spikes_t = poisson_spiking_gen(pre_rate, post_rate, t_run, dt, noise)

	# Brian2's NeuronGroup
	Pre, Post = load_neurons(N_Pre, N_Post, neuron_type,
		spikes_t_Pre = pre_spikes_t,
		spikes_t_Post = post_spikes_t,
		pre_rate = pre_rate,
		post_rate =  post_rate)