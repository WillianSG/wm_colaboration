# -*- coding: utf-8 -*-
"""
@author: wgirao
@based-on: asonntag

Comments:
- https://brian2.readthedocs.io/en/stable/resources/tutorials/2-intro-to-brian-synapses.html
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

from matplotlib import *
from matplotlib import pyplot as plt
import matplotlib as mpl

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
experiment_type = 'single_synapse'

# Simulation
dt_resolution = 0.00001 # = 0.00001 sconds (0.01ms) | simu. time resolution
exp_type = 'stdp_curve_'

tmax = 100*ms # maximal difference between spike timing
duration = 2*tmax # simulation duration

# Brian
t_run = 0.200 # (s)

neuron_type = 'spikegenerator' # used only to load synapse model

N_Pre = 50 # neurons
N_Post = 50
N_t = 50

plasticity_rule = 'LR1'
parameter_set = '1.1'

int_meth_syn = 'euler' # Synaptic integration method

# obj holding the simu. time and the time step
defaultclock.dt = dt_resolution*second # time step of the simu. as a float

# 2 ========== Network setup ==========

# Loading rule's parameter set
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

# Loading model equations
[model_E_E,
	pre_E_E,
	post_E_E] = load_synapse_model(plasticity_rule, neuron_type, False)

print("\n> Model loaded")

# Setting network objects
Pre = NeuronGroup(N_Pre, 'tspike:second', threshold='t>tspike', 
	refractory = 1000*second)

Post = NeuronGroup(N_Post, 'tspike:second', threshold='t>tspike', 
	refractory = 1000*second)

Pre.tspike = 'i*tmax/(N_t-1)' # spike timings of presynaptic neurons
Post.tspike = '(N_t-1-i)*tmax/(N_t-1)' # spike timings of postsynaptic neurons

Pre_Post = Synapses(Pre, Post, model = model_E_E,
on_pre = pre_E_E,
on_post = post_E_E,
method = int_meth_syn,
name = 'Pre_Post')

Pre_Post.connect(j='i')

Pre_Post.rho = rho_init

num_Pre_Post_synapses = len(Pre_Post.i)

# Monitors
Pre_mon = SpikeMonitor(Pre)
Post_mon = SpikeMonitor(Post)
Pre_Post_rec = StateMonitor(Pre_Post, ('rho', 'xpre', 'xpost'),
	record = True)

print("\n> Network objs loaded")

# Running simulation
print("\nRunning simulation...")
run(duration)

# 3 ========== Plotting results ==========

# Results dir
results_path = os.path.join(parent_dir, 'plots_results')
is_dir = os.path.isdir(results_path)

if not(is_dir):
	os.mkdir(results_path)

file_name = exp_type + plasticity_rule + "_" + parameter_set + "_" + sim_id

plots_dir = os.path.join(results_path, file_name)
is_dir = os.path.isdir(plots_dir)

if not(is_dir):
	os.mkdir(plots_dir)

s1 = 35
lwdth = 5
mpl.rcParams['axes.linewidth'] = lwdth/2

fig = plt.figure(figsize=(25, 17.5))

plt.axhline(0, ls = '-', c='grey', lw = lwdth)
plt.axvline(0, ls = '-', color='grey', lw = lwdth)

dw = Pre_Post.rho-(ones(len(Pre_Post.rho))*rho_init) # delta_w at each update

plt.plot((Post.tspike-Pre.tspike)/ms, dw,'-o', 
	color = 'lightsteelblue',
	label = '%s    %s' %(tau_xpre,np.around(rho_neg, 3)), 
	lw = lwdth, 
	markersize = 20)

plt.xlabel('$\Delta t$ (ms)', fontsize = s1)
plt.ylabel('Synaptic efficacy\nchange $\Delta \\rho$',
	horizontalalignment='center', fontsize = s1)

plt.xlim(-tmax/ms*1.05, tmax/ms*1.05)
plt.xticks(np.arange(-tmax/ms, tmax/ms+1,20), rotation = 45)
plt.tick_params(axis = 'both', which = 'major', length = 5)

plt.tick_params(axis = 'both', which = 'major', width = lwdth, 
	length = 9,
	labelsize = s1, 
	pad = 10)

plt.legend(title = 'Parameter: \n  $\\tau_{xpre}$:     $\\rho_{neg}$:',
	title_fontsize = s1,
	loc ='upper left', 
	fontsize = s1)

plt.savefig(os.path.join(plots_dir, 
	file_name + ".png"),
	bbox_inches = 'tight',
	dpi = 200)

print("\n> Results ploted to file")

print('\nrun_stdp_curve.py - END.\n')