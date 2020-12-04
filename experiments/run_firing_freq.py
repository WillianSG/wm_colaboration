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
from run_frequencies import *

# 1 ========== Execution parameters ==========

# Simulation run variables
dt_resolution = 0.1/1000 # = 0.0001 sconds (0.1ms) | step of simulation time step resolution
t_run = 5 # simulation time (seconds)
noise = 0.75 # used to introduce difference between spike times betweem pre- and post-

N_Pre = 1
N_Post = 1

isi_correlation = 'negative' # "random", "positive", "negative"
plasticity_rule = 'LR2' # 'none', 'LR1', 'LR2'
parameter_set = '2.2' # '2.1'
neuron_type = 'spikegenerator' # 'poisson', 'LIF' , 'spikegenerator'
bistability = True
drho_all_metric = 'original' # 'original', 'mean'

exp_type = 'firing_freq_parallel_'+isi_correlation

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
def run_net_parallel(p, q):
	print('pre @ ', pre_freq[p], 'Hz, post @ ', post_freq[q], 'Hz')

	ans = run_frequencies(pre_freq[p], post_freq[q], t_run, dt_resolution, plasticity_rule, neuron_type, noise, bistability, plot_single_trial, N_Pre, N_Post, tau_xpre, tau_xpost, xpre_jump, xpost_jump, rho_neg, rho_neg2, rho_init, tau_rho, thr_post, thr_pre, thr_b_rho, rho_min, rho_max, alpha, beta, xpre_factor, w_max, model_E_E, pre_E_E, post_E_E, int_meth_syn, isi_correlation, drho_all_metric, job_seed)

	return p, q, ans

# check if running on cluster and adapt the multithread exec to cluster job
if "SLURM_CPUS_ON_NODE" in os.environ:
	num_cores = int(os.environ['SLURM_CPUS_ON_NODE'])
	print("\n# cluster job cores: ", num_cores)
else:
	num_cores = multiprocessing.cpu_count()
	print("\n# machine cores: ", num_cores)


print("Running network...\n")

# Running network for each pair of frequency
results = Parallel(n_jobs=num_cores)(delayed(run_net_parallel)(p,q) for p,q in simulationsset)

for t in results:
	p = t[0] # x
	q = t[1] # y
	final_rho_all[p,q], drho_all[p,q] = t[2] # '(last rho, last rho/initial rho)' of each execution, for (p,q) pairs of firing ratesd

## Saving results + metadata
# distingish between cluster and local exec
if "SLURM_ARRAY_JOB_ID" in os.environ:
	path_sim_id = os.path.join(results_path, str(os.environ['SLURM_ARRAY_JOB_ID']) + '_' + exp_type)
else:
	path_sim_id = os.path.join(results_path, sim_id +'_' + exp_type)

if not os.path.exists(path_sim_id):
	os.mkdir(path_sim_id)

fn = str(job_seed) + '_' + exp_type + '_w_final_drho.pickle'

fnopen = os.path.join(path_sim_id, fn)

with open(fnopen,'wb') as f:
	pickle.dump((
		final_rho_all,
		drho_all,
		pre_freq,
		post_freq,
		min_freq,
		max_freq,
		step,
		sim_id,
		exp_type,
		plasticity_rule,
		parameter_set,
		neuron_type,
		bistability,
		dt_resolution,
		t_run,
		noise,
		N_Pre,
		N_Post,
		int_meth_syn,
		isi_correlation,
		drho_all_metric,
		job_seed)
		, f)

print('\nrun_firing_freq.py - END.\n')
