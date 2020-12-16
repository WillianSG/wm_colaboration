# -*- coding: utf-8 -*-
"""
@author: wgirao
@based-on: asonntag
"""
import setuptools
import os, sys, pickle
import platform
from brian2 import *
from scipy import *
from numpy import *
from joblib import Parallel, delayed
from time import localtime
import multiprocessing
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

# Results dir check
results_path = os.path.join(parent_dir, 'sim_results')

is_dir = os.path.isdir(results_path)
if not(is_dir):
	os.mkdir(results_path)

# Creating simulation ID
idt = localtime()
sim_id = str(idt.tm_year) \
	+ '{:02}'.format(idt.tm_mon) \
	+ '{:02}'.format(idt.tm_mday) + '_' \
	+ '{:02}'.format(idt.tm_hour) + '_' \
	+ '{:02}'.format(idt.tm_min)

# Starts a new scope for magic functions
start_scope()

# Helper modules
from load_parameters import *
from load_synapse_model import *
from load_neurons import *

# 1 ========== Execution parameters ==========

exp_type = 'showcase'

# Simulation run variables
dt_resolution = 0.1/1000 # = 0.0001 sconds (0.1ms) | step of simulation time step resolution

t_run = 3 # simulation time (seconds)

int_meth_syn = 'euler' # Synaptic integration method

# 1.1 ========== Rule's parameters

plasticity_rule = 'LR1' # 'none', 'LR1', 'LR2'
parameter_set = '1.2' # '2.1'
bistability = True

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

w_init = w_max*rho_init

# 1.2 ==== net parameters

N_Pre = 1
N_Post = 1

pre_rate = 80
post_rate = 20 

if exp_type == 'showcase':
	neuron_type = 'spikegenerator'
else:
	neuron_type = 'poisson'


# 2 ========== Learning rule as Brian2's synaptic model
[model_E_E,
	pre_E_E,
	post_E_E] = load_synapse_model(plasticity_rule, neuron_type, bistability)



# 3 ========== Brian2's neuron objects

input_pre = np.array([10, 60, 165, 255])/1000
input_post = np.array([25, 65, 140, 250])/1000 

Pre, Post = load_neurons(
	N_Pre, N_Post, neuron_type,
	spikes_t_Pre = input_pre,
	spikes_t_Post = input_post,
	pre_rate = pre_rate,
	post_rate =  post_rate)

Pre_Post = Synapses(
	source = Pre,
	target = Post,
	model = model_E_E, 
	on_pre = pre_E_E,
	on_post = post_E_E,
	method = int_meth_syn, 
	name = 'Pre_Post')

Pre_Post.connect(j = 'i')  

Pre_Post.rho = rho_init
Pre_Post.w = w_init

StateMon = StateMonitor(Pre_Post, ['xpre', 'xpost', 'w', 'rho'], record = True)

Pre_spk_mon = SpikeMonitor( 
	source = Pre,
	record = True,
	name = 'Pre_spk_mon')

Post_spk_mon = SpikeMonitor( 
	source = Post,
	record = True,
	name = 'Post_spk_mon')

run(t_run*second)


print('\nnet_single_synapse.py - END.\n')