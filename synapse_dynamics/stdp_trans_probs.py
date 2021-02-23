# -*- coding: utf-8 -*-
"""
@author: wgirao

Input:

Output:

Comments:
- Run single synapse simulations to get transition probabilities for LTP and LTD.
"""
import setuptools
import os, sys
from brian2 import *
from scipy import *
from numpy import *

prefs.codegen.target = 'numpy'

helper_dir = 'helper_functions'

# get run id as seed for random gens
try:
	job_seed = int(sys.argv[1])
except:
	job_seed = int(0)

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))

# Helper modules
from load_parameters import *
from load_synapse_model import *
from run_single_synap import *

# == 1 - Simulation run variables ==========

dt_resolution = 0.0001 # 0.1ms | step of simulation time step resolution
t_run = 0.3 # 300ms | simulation time (seconds)
noise = 0.75 # introduced noise (diff) between spk times betweem pre- and post-

N_Pre = 1
N_Post = 1

plasticity_rule = 'LR2'			# 'none', 'LR1', 'LR2'
parameter_set = '2.4'			# '2.1', '2.4'
bistability = False

neuron_type = 'spikegenerator'	# 'poisson', 'LIF' , 'spikegenerator'
int_meth_syn = 'euler'			# Synaptic integration method

exp_type = 'stdp_trans_probabi'

# Range of pre- and postsynaptic frequencies (Hz)
min_freq = 0
max_freq = 60
step = 5

# == 1.1 - neurons' firing rates

f_pre = np.arange(min_freq, max_freq + 0.1, step)
f_pos = np.arange(min_freq, max_freq + 0.1, step)

# == 2 - Brian2 simulation settings ==========

# loading learning rule parameters
[tau_xpre,
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
	w_max] = load_rule_params(plasticity_rule, parameter_set)

# loading synaptic rule equations
[model_E_E,
	pre_E_E,
	post_E_E] = load_synapse_model(plasticity_rule, neuron_type, bistability)



# Starts a new scope for magic functions
start_scope()

result = run_single_synap(
	pre_rate = f_pre[1], 
	post_rate = f_pos[1], 
	t_run = t_run, 
	dt_resolution = dt_resolution, 
	plasticity_rule = plasticity_rule, 
	neuron_type = neuron_type, 
	noise = noise, 
	bistability = bistability, 
	N_Pre = N_Pre, 
	N_Post = N_Post, 
	tau_xpre = tau_xpre, 
	tau_xpost = tau_xpost, 
	xpre_jump = xpre_jump, 
	xpost_jump = xpost_jump, 
	rho_neg = rho_neg, 
	rho_neg2 = rho_neg2, 
	rho_init = rho_init, 
	tau_rho = tau_rho, 
	thr_post = thr_post, 
	thr_pre = thr_pre, 
	thr_b_rho = thr_b_rho, 
	rho_min = rho_min, 
	rho_max = rho_max, 
	alpha = alpha, 
	beta = beta, 
	xpre_factor = xpre_factor, 
	w_max = w_max, 
	model_E_E = model_E_E, 
	pre_E_E = pre_E_E, 
	post_E_E = post_E_E, 
	int_meth_syn = int_meth_syn,
	job_seed = job_seed)

print('\n final efficacy (w): ', result)


# for f_pos in range(0, len(post_freq)):
# 	for f_pre in range(0, len(pre_freq)):
# 		print('post @ ', post_freq[f_pos], 'Hz, pre @ ', pre_freq[f_pre], 'Hz')
# 	print('\n')