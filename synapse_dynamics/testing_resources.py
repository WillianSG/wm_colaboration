# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System

Function:
- Just shows the dynamics of the syn variables x and u.

To Do:
- [] post as LIF neuron to test 'Vepsp_transmission_LR4 = {'Vepsp_transmission' : '''Vepsp += w*(u*x_)'''}'
"""
import setuptools
import os, sys, pickle
import platform
from brian2 import *
from scipy import *
from numpy import *
from joblib import Parallel, delayed
from time import localtime, strftime
import multiprocessing

prefs.codegen.target = 'numpy'

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors

pre_rate = 0
post_rate = 0
num_sim = 1
rule = 'LR4'
parameter_set = '1.0'

# 0 ========== User degined imports ==========

helper_dir = 'helper_functions'

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))

# Results dir check
results_path = os.path.join(parent_dir, 'synapse_results')

is_dir = os.path.isdir(results_path)
if not (is_dir):
    os.mkdir(results_path)

# Creating simulation ID
idt = localtime()
sim_id = strftime("%d%b%Y_%H-%M-%S_", localtime())

# Starts a new scope for magic functions
start_scope()

# Helper modules
from helper_functions.load_rule_parameters import *
from helper_functions.load_synapse_model import *
from helper_functions.load_neurons import *

# 1 ========== Execution parameters ==========

exp_type = 'showcase'  # 'showcase', 'rates'
dt_resolution = 0.001  # secs | simulation step time resolution
t_run = 1  # simulation time (seconds)
int_meth_syn = 'euler'  # Synaptic integration method

N_Pre = 1  # presynaptic neurons
N_Post = 1  # postsynaptic neurons

# 1.1 ========== Rule's parameters

plasticity_rule = rule  # 'none', 'LR1', 'LR2', 'LR3'
bistability = True
stoplearning = False
resources = True

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
	w_max,
	xpre_min,
	xpost_min,
	xpost_max,
	xpre_max,
	tau_xstop,
	xstop_jump,
	xstop_max,
	xstop_min,
	thr_stop_h,
	thr_stop_l,
	U,
	tau_d,
	tau_f] = load_rule_parameters(
		plasticity_rule = plasticity_rule, 
		parameter_set = parameter_set)

w_init = w_max * rho_init

# 1.2 ========== net parameters

if exp_type == 'showcase':
    neuron_type = 'spikegenerator'
else:
    neuron_type = 'poisson'

# 2 ========== Learning rule as Brian2's synaptic model ==========
[model_E_E,
 pre_E_E,
 post_E_E] = load_synapse_model(
    plasticity_rule,
    neuron_type,
    bistability,
    stoplearning=stoplearning,
    resources=resources)

# 3 ========== Brian2's neuron objects

# only used when 'exp_type' is set to 'showcase'
input_pre = np.array([100, 140, 180, 220, 260, 300, 340, 540]) / 1000
# input_post = np.array([15, 125, 180, 200, 300]) / 1000
input_post = np.array([])

Pre, Post = load_neurons(
    N_Pre, N_Post, neuron_type,
    spikes_t_Pre=input_pre,
    spikes_t_Post=input_post,
    pre_rate=pre_rate,
    post_rate=post_rate)

# 3.1 ========== setting connections between neurons
Pre_Post = Synapses(
    source=Pre,
    target=Post,
    model=model_E_E,
    on_pre=pre_E_E,
    on_post=post_E_E,
    method=int_meth_syn,
    name='Pre_Post')

Pre_Post.connect(j='i')

Pre_Post.x_ = 1.0  # so that it starts with value 1.0
Pre_Post.u = U  # so that it starts with value U

Pre_Post.plastic = True
Pre_Post.rho = rho_init
Pre_Post.w = w_init

# 3.2 ========== Setting simulation monitors

StateMon = StateMonitor(Pre_Post, ['xpre', 'xpost', 'w', 'rho', 'x_', 'u'], record=True)

Pre_spk_mon = SpikeMonitor(
    source=Pre,
    record=True,
    name='Pre_spk_mon')

Post_spk_mon = SpikeMonitor(
    source=Post,
    record=True,
    name='Post_spk_mon')

run(t_run * second)

fig = plt.figure(constrained_layout=True, figsize=(5, 8))

widths = [8]
heights = [0.25, 2.5, 0.25]

spec2 = gridspec.GridSpec(
    ncols=1,
    nrows=3,
    width_ratios=widths,
    height_ratios=heights,
    figure=fig)

ax1 = fig.add_subplot(spec2[0, 0])

y = [0] * len(Pre_spk_mon.t)
plt.plot(Pre_spk_mon.t / ms, y, 'k|')
plt.xlabel('Time (ms)')
plt.ylabel('pre spikes')

ax2 = fig.add_subplot(spec2[1, 0])

plt.plot(StateMon.t / ms, StateMon.x_[0], label='x', color='tomato')
plt.plot(StateMon.t / ms, StateMon.u[0], label='u', color='blue')
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('synaptic variables')

ax3 = fig.add_subplot(spec2[2, 0])

y = [0] * len(Post_spk_mon.t)
plt.plot(Post_spk_mon.t / ms, y, 'k|')
plt.xlabel('Time (ms)')
plt.ylabel('post spikes')

fig.suptitle(
    'rule: ' + str(rule) + ' | parameters: ' + str(parameter_set),
    fontsize=10)

plt.show()