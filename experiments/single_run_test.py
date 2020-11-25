# -*- coding: utf-8 -*-
"""
@author: wgirao
@based-on: asonntag

single net run test
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
from brian2 import Synapses,StateMonitor, SpikeMonitor, defaultclock, Network, second

helper_dir = 'helper_functions'

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))

from poisson_spiking_gen import *
from load_neurons import *

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

# Helper modules
from load_parameters import *
from load_synapse_model import *
from run_frequencies import *

# 1 ========== Execution parameters ==========

# Simulation run variables
dt_resolution = 0.1/1000 # = 0.0001 sconds (0.1ms) | step of simulation time step resolution
t_run = 5 # simulation time (seconds)
noise = 0.75 # used to introduce difference between spike times betweem pre- and post-

N_Pre = 1
N_Post = 1

exp_type = 'firing_freq_parallel'
isi_correlation = 'random'
plasticity_rule = 'LR2' # 'none', 'LR1', 'LR2'
parameter_set = '2.2' # '2.1'
neuron_type = 'poisson' # 'poisson', 'LIF' , 'spikegenerators'
bistability = True

int_meth_syn = 'euler' # Synaptic integration method

# Plotting settings
plot_single_trial = False  # True = plot single simulations

# Range of pre- and postsynaptic frequencies (Hz)
min_freq = 0
max_freq = 5
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

# Starts a new scope for magic functions
start_scope()

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

# 2 ========== Running network in parallel ==========
# def run_net_parallel(p, q):
# 	print('pre @ ', p, 'Hz, post @ ', q, 'Hz')

# 	ans = run_frequencies(pre_freq[p], post_freq[q], t_run, dt_resolution, plasticity_rule, neuron_type, noise, bistability, plot_single_trial, N_Pre, N_Post, tau_xpre, tau_xpost, xpre_jump, xpost_jump, rho_neg, rho_neg2, rho_init, tau_rho, thr_post, thr_pre, thr_b_rho, rho_min, rho_max, alpha, beta, xpre_factor, w_max, model_E_E, pre_E_E, post_E_E, int_meth_syn, isi_correlation)

# 	return p, q, ans


# Spike time arrays
pre_spikes_t, post_spikes_t = poisson_spiking_gen(
	rate_pre = pre_freq[1], 
	rate_post = post_freq[1], 
	t_run = t_run, 
	dt = dt_resolution, 
	noise = noise,
	correlation = isi_correlation)

# Brian2's NeuronGroup
Pre, Post = load_neurons(
	N_Pre, N_Post, neuron_type,
	spikes_t_Pre = pre_spikes_t,
	spikes_t_Post = post_spikes_t,
	pre_rate = pre_freq[1],
	post_rate =  post_freq[1])

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


net.run(t_run*second) # simulating (running network)

# Simulation data
"""
spike_trains() - dictionary with keys being the indices of the neurons and values being the array of spike times for that neuron
"""
# - neurons related
spiketrains_PRE = Pre_mon.spike_trains()[0]
spiketrains_POST = Post_mon.spike_trains()[0]

# - synapse related (all array lenths are equal)
efficacy_PRE_POST = synaptic_mon.rho[0][:] # Rho variable evolution
weight_PRE_POST = synaptic_mon.w[0][:] # Weight (w) variable evolution
rec_t_PRE_POST = synaptic_mon.t[:] # Simulations time steps

# - store final w value and calculate dw
final_rho_all = Pre_Post.rho[0] # last value of 'rho' at the end of the ex.
drho_all = Pre_Post.rho[0] / rho_init # synaptic weight change (last/init)

print("\n Rho: ")
print(synaptic_mon.rho[0], "\n\n" , Pre_Post.rho[0], "\n\n\n\n\n\n\n\n")
print(mean(synaptic_mon.rho[0]))


print('\nsingle_run_test.py - END.\n')