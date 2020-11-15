# -*- coding: utf-8 -*-
"""
@author: wgirao
@based-on: asonntag
"""
import setuptools
import os, sys
import platform
from brian2 import *
from scipy import *
from numpy import *
prefs.codegen.target = "numpy"

helper_dir = 'helper_functions'

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))

# Helper modules
from load_parameters import *
from load_synapse_model import *
from run_frequencies import *

# 1 ========== Experiment parameters ==========

# Simulation run variables
dt = 0.1/1000 # = 0.0001 sconds (0.1ms) | step of simulation time step resolution
t_run = 5 # time of simulation (seconds ?)
noise = 0.75 # ?

N_Pre = 1
N_Post = 1

plasticity_rule = 'LR2' # 'none', 'LR1', 'LR2'
parameter_set = '2.1' # '2.1'
neuron_type = 'poisson' # 'poisson', 'LIF' , 'spikegenerators'
bistability = False

# Plotting settings
plot_single_trial = False  # True = plot single simulations

# Range of pre- and postsynaptic frequencies
min_freq = 0
max_freq = 100
step = 5

# Frequency activity ranges (for pre and post neurons)
pre_freq = np.arange(min_freq, max_freq+0.1, step)
post_freq = np.arange(min_freq, max_freq+0.1, step)

# Heat-map's (x,y) coordinates
simulationsset = []
for p in np.arange(0,len(pre_freq),1):
    for q in np.arange(0,len(post_freq),1):
        simulationsset.append((p,q))

# Empty containers for final rho value and drho = rho_final / rho_init
# each square of the heatmap - thus np.zeros(pref,postf)
final_rho_all = np.zeros((len(pre_freq),len(post_freq)))
drho_all = np.zeros((len(pre_freq),len(post_freq)))

# 2 ========== Rule's parameters ==========
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

# 2.1 ========== Learning rule as Brian2's synaptic model
[model_E_E, 
	pre_E_E, 
	post_E_E] = load_synapse_model(plasticity_rule, neuron_type, bistability)


# ========== Brian stuff ==========

ans = run_frequencies(pre_freq[3], post_freq[1], t_run, dt, plasticity_rule, neuron_type, noise, bistability, plot_single_trial, N_Pre, N_Post, tau_xpre, tau_xpost, xpre_jump, xpost_jump, rho_neg, rho_neg2, rho_init, tau_rho, thr_post, thr_pre, thr_b_rho, rho_min, rho_max, alpha, beta, xpre_factor, w_max, model_E_E, pre_E_E, post_E_E)


# Starts a new scope for magic functions
start_scope()

print('END\n')