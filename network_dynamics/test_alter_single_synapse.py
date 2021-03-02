# -*- coding: utf-8 -*-
"""
@author: wgirao

Input:

Output:

Comments:
- Run single synapse simulations to get transition probabilities for LTP and LTD.
- [REFACTOR]: use multithreading for frequencies loop.
"""
import setuptools
import os, sys, pickle
from brian2 import *
from scipy import *
import numpy as np

from matplotlib.pyplot import *

prefs.codegen.target = 'numpy'

helper_dir = 'helper_functions'

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))

# Helper modules
from load_parameters import *
from load_synapse_model import *
from run_single_synap import *

# == 1 - Simulation run variables ==========

dt_resolution = 0.0001		# 0.1ms | step of simulation time step resolution
t_run = 1					# simulation time (seconds)

N_Pre = 3
N_Post = 3

plasticity_rule = 'LR2'			# 'none', 'LR1', 'LR2'
parameter_set = '2.2'			# '2.1', '2.2', '2.4'
bistability = True

neuron_type = 'spikegenerator'	# 'poisson', 'LIF' , 'spikegenerator'
int_meth_syn = 'euler'			# Synaptic integration method

exp_type = 'stdp_trans_probabi_'

# Range of pre- and postsynaptic frequencies (Hz)
min_freq = 0
max_freq = 100

step = 5

# == 1.1 - neurons' firing rates

f_pre = np.arange(min_freq, max_freq + 0.1, step)
f_pos = np.arange(min_freq, max_freq + 0.1, step)

resul_per_pre_rate = []
transition_probabilities = []

# == 2 - Brian2 simulation settings ==========

# Starts a new scope for magic functions
start_scope()

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
	w_max] = load_rule_params(
		plasticity_rule = plasticity_rule, 
		parameter_set = parameter_set)

# loading synaptic rule equations
[model_E_E,
	pre_E_E,
	post_E_E] = load_synapse_model(plasticity_rule, neuron_type, bistability)

Pre = PoissonGroup(
	N = N_Pre,
	rates = 5*Hz)

Post = PoissonGroup(
	N = N_Post,
	rates = 5*Hz)

Pre_Post = Synapses(
	source = Pre,
	target = Post,
	model = model_E_E,
	on_pre = pre_E_E,
	on_post = post_E_E,
	method = int_meth_syn,
	name = 'Pre_Post')

Pre_Post.connect(i = [0, 0, 0, 1, 2], j = [0, 1, 2, 1, 2])

# Initializating value to all synapses
Pre_Post.rho = rho_init

# creating matrix from connections
W = np.full((len(Pre), len(Post)), np.nan)
W[Pre_Post.i[:], Pre_Post.j[:]] = Pre_Post.rho[:]

for pre_id in range(0, len(Pre)):
	for post_id in range(0, len(Post)):
		if np.isnan(W[pre_id][post_id]) == False:
			print(pre_id, post_id, ' = ', Pre_Post.rho[pre_id , post_id])

print('\n')

# ========== altering single synapses ==========


for pre_id in range(0, len(Pre)):
	for post_id in range(0, len(Post)):
		if isnan(W[pre_id][post_id]) == False:
			sigma = (rho_init*15)/100					# standard deviation
			s = np.random.normal(rho_init, sigma, 1)	# sampling new val.

			Pre_Post.rho[pre_id , post_id] = s[0]

for pre_id in range(0, len(Pre)):
	for post_id in range(0, len(Post)):
		if isnan(W[pre_id][post_id]) == False:
			print(pre_id, post_id, ' = ', Pre_Post.rho[pre_id , post_id])

# print('init: ', rho_init)

# print('i0 j0: ', Pre_Post.rho[0 , 0])
# print('i1 j1: ', Pre_Post.rho[1 , 1])

print('\n')

W = np.full((len(Pre), len(Post)), np.nan)
W[Pre_Post.i[:], Pre_Post.j[:]] = Pre_Post.rho[:]

print(W)

# for pre_id in range(0, len(Pre)):
# 	for post_id in range(0, len(Post)):
# 		if isnan(W[pre_id][post_id]) == False:
# 			print(pre_id, post_id)

# ========== plotting ==========

Ns = len(Pre_Post.source)
Nt = len(Pre_Post.target)

figure(figsize=(10, 4))

subplot(121)

plot(zeros(Ns), arange(Ns), 'ok', ms=10)
plot(ones(Nt), arange(Nt), 'ok', ms=10)

for i, j in zip(Pre_Post.i, Pre_Post.j):
    plot([0, 1], [i, j], '-k')

xticks([0, 1], ['Source', 'Target'])
ylabel('Neuron index')
xlim(-0.1, 1.1)
ylim(-1, max(Ns, Nt))
subplot(122)
plot(Pre_Post.i, Pre_Post.j, 'ok')
xlim(-1, Ns)
ylim(-1, Nt)
xlabel('Source neuron index')
ylabel('Target neuron index')

show()
