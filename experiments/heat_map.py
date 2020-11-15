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

# 1 ========== Experiment parameters ==========

plasticity_rule = 'LR2' # 'none', 'LR1', 'LR2'
parameter_set = '2.1' # '2.1'
neuron_type = 'poisson' # 'poisson', 'LIF' , 'spikegenerators'
bistability = False

# Range of pre- and postsynaptic frequencies
min_freq = 0
max_freq = 100
step = 5

# Frequencies (x and y on coords.)
pre_freq = np.arange(min_freq, max_freq+0.1, step)
post_freq = np.arange(min_freq, max_freq+0.1, step)

# Empty containers for final rho value and drho = rho_final / rho_init
# each square of the heatmap - thus np.zeros(pref,postf)
final_rho_all = np.zeros((len(pre_freq),len(post_freq)))
drho_all = np.zeros((len(pre_freq),len(post_freq)))

# 2 ========== Rule's parameters ==========
[tau_xpre,
	tau_xpost,
	xpre_jump,
	xpost_jump,
	rho_dep,
	rho_dep2,
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


# ========== Brian stuff ==========
[model_E_E, 
	pre_E_E, 
	post_E_E] = load_synapse_model(plasticity_rule, neuron_type, bistability)


# Starts a new scope for magic functions
start_scope()

print('END\n')