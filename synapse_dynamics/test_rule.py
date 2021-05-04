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
from time import localtime, strftime
import multiprocessing
prefs.codegen.target = 'numpy'

#=====
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors
#=====

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
sim_id = strftime("%d%b%Y_%H-%M-%S_", localtime())

# Starts a new scope for magic functions
start_scope()

# Helper modules
from load_parameters import *
from load_synapse_model import *
from load_neurons import *

# 1 ========== Execution parameters ==========

exp_type = 'rates' # 'showcase', 'rates'

# Simulation run variables
dt_resolution = 0.001 # = 0.0001 sconds (0.1ms) | step of simulation time step resolution

t_run = 1 # simulation time (seconds)

int_meth_syn = 'euler' # Synaptic integration method

# 1.1 ========== Rule's parameters

plasticity_rule = 'LR3' # 'none', 'LR1', 'LR2'
parameter_set = '4.0'
bistability = False

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
	xpre_max] = load_rule_params(plasticity_rule, parameter_set)

w_init = w_max*rho_init

# 1.2 ========== net parameters

N_Pre = 1
N_Post = 1

pre_rate = int(sys.argv[1])
post_rate = int(sys.argv[2])

if exp_type == 'showcase':
	neuron_type = 'spikegenerator'
else:
	neuron_type = 'poisson'

# 2 ========== Learning rule as Brian2's synaptic model ==========
[model_E_E,
	pre_E_E,
	post_E_E] = load_synapse_model(plasticity_rule, neuron_type, bistability)

# 3 ========== Brian2's neuron objects

input_pre = np.array([10, 15, 18, 25, 27, 29, 31, 32])/1000
input_post = np.array([25, 65, 105, 250])/1000

Pre, Post = load_neurons(
	N_Pre, N_Post, neuron_type,
	spikes_t_Pre = input_pre,
	spikes_t_Post = input_post,
	pre_rate = pre_rate,
	post_rate =  post_rate)

# 3.1 ========== setting connections between neurons
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

# 3.2 ========== Setting simulation monitors

StateMon = StateMonitor(Pre_Post, ['xpre', 'xpost', 'w', 'rho'], record = True)

Pre_spk_mon = SpikeMonitor( 
	source = Pre,
	record = True,
	name = 'Pre_spk_mon')

Post_spk_mon = SpikeMonitor( 
	source = Post,
	record = True,
	name = 'Post_spk_mon')

# 4. ========== Running network ==========

run(t_run*second)

# 5. ========== Plots ==========

lwdth = 3

s1 = 30

mpl.rcParams['axes.linewidth'] = 1.5

# Width, height in inches.
fig = plt.figure(figsize = (15, 26))

gs = gridspec.GridSpec(12, 1, height_ratios = [2, 2, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4])

# 5.1 ==== Pre neuron spike activity

ax1 = fig.add_subplot(gs[0, 0])
ax1.vlines(Pre_spk_mon.t/ms, 0.5, 1.75, color = 'blue', linewidth = lwdth)

ax1.axhline(linewidth = lwdth, color = 'blue', y = 0.5)
ax1.set_yticks([])
ax1.set_xticklabels([])
plt.ylabel('Pre ${i}$', fontsize = s1, color = 'black', rotation = 0)
ax1.yaxis.set_label_coords(-0.15, 0.3)
plt.tick_params(axis = 'x', which = 'major', width = lwdth, length = 5)
plt.tick_params(axis = 'y', which = 'major', width = lwdth, length = 0)
plt.ylim(0, 2)
plt.xlim(0, t_run*1000)
plt.xticks(size = s1)
plt.yticks(size = s1)

# 5.2 ==== Post neuron spike activity

ax2 = fig.add_subplot(gs[1, 0])
ax2.vlines(Post_spk_mon.t/ms, 0.5, 1.75, color = 'red', linewidth = lwdth)
ax2.axhline(linewidth = lwdth, color = 'red', y = 0.5)
ax2.set_yticks([])
ax2.set_xticklabels([])

plt.ylabel('Post ${j}$', fontsize = s1, color = 'black', rotation = 0)

ax2.yaxis.set_label_coords(-0.15, 0.3)

plt.tick_params(axis = 'x', which = 'major', width = lwdth, length = 5)
plt.tick_params(axis = 'y', which = 'major', width = lwdth, length = 0)
plt.ylim(0, 2)
plt.xlim(0, t_run*1000)
plt.xticks(size = s1)
plt.yticks(size = s1)

# 5.3 ==== Presynaptic calcium trace

ax3 = fig.add_subplot(gs[3, 0])
ax3.plot(StateMon.t/ms, StateMon.xpre[0], color = 'blue', linewidth = lwdth)

# Adapted learning rule has threshold on pre- trace as well
if plasticity_rule == 'LR2' or plasticity_rule == 'LR3':
	ax3.axhline(linestyle = 'dashed', color = 'grey', lw = lwdth/2, 
		y = thr_pre, 
		label = '$\\theta_{pre}$')

	plt.legend(loc = 'upper right', prop = {'size':s1-10}, 
		bbox_to_anchor = (1, 1.3), 
		ncol = 3)

plt.ylabel('$x_{Pre}$ \n(a.u.)', size = s1, color = 'black',
	horizontalalignment = 'center', 
	labelpad = 20)

ax3.yaxis.set_label_coords(-0.15, 0.3)
ax3.set_xticklabels([])

plt.tick_params(axis = 'x', which = 'major', width = lwdth, length = 5)
plt.tick_params(axis = 'y', which = 'major', width = lwdth, length = 0)

major_yticks = np.linspace(0, max(StateMon.xpre[0])*1.1, 4)

ax3.set_yticks(np.around(major_yticks, 1))

plt.ylim(0, max(StateMon.xpre[0])*1.1)
plt.xlim(0, t_run*1000)
plt.xticks(size = s1)
plt.yticks(size = s1)

# 5.4 ==== Postsynaptic calcium trace

ax4 = fig.add_subplot(gs[5, 0])
ax4.plot(StateMon.t/ms, StateMon.xpost[0], color = 'red', linewidth = lwdth) 
ax4.axhline(linestyle = 'dashed', color = 'grey', lw = lwdth/2, y = thr_post,
	label = '$\\theta_{post}$')
ax4.set_xticklabels([])

plt.legend(loc = 'upper right', prop = {'size':s1-10}, 
	bbox_to_anchor = (1, 1.3), 
	ncol = 3)

plt.ylabel('$x_{Post}$ \n(a.u.) ', size = s1, color = 'black', 
	horizontalalignment = 'center', 
	labelpad = 20)

ax4.yaxis.set_label_coords(-0.15, 0.3)
ax4.set_xticklabels([])

plt.tick_params(axis = 'x', which = 'major', width = lwdth, length = 5)
plt.tick_params(axis = 'y', which = 'major', width = lwdth, length = 0)

major_yticks = np.linspace(0, max(StateMon.xpost[0])*1.1, 4)

ax4.set_yticks(np.around(major_yticks, 1))

plt.ylim(0, np.around(max(StateMon.xpost[0]), 1)*1.1)
plt.xlim(0, t_run*1000)
plt.xticks(size = s1)
plt.yticks(size = s1)

# 5.5* ==== Stop-learning calcium trace

if plasticity_rule == 'LR3_1' or plasticity_rule == 'LR3_2':
	ax4 = fig.add_subplot(gs[7, 0])
	ax4.plot(StateMon.t/ms, StateMon.xstop[0], color = 'tab:blue', linewidth = lwdth) 

	ax4.axhline(linestyle = 'solid', color = 'grey', lw = lwdth/2, y = thr_stop_h,
		label = '$\\theta_{stop}^{h}$')

	ax4.axhline(linestyle = 'dotted', color = 'grey', lw = lwdth/2, y = thr_stop_l,
		label = '$\\theta_{stop}^{l}$')

	ax4.set_xticklabels([])

	plt.legend(loc = 'upper right', prop = {'size':s1-10}, 
		bbox_to_anchor = (1, 1.3), 
		ncol = 4)

	plt.ylabel('$x_{stop}$ \n(a.u.) ', size = s1, color = 'black', 
		horizontalalignment = 'center', 
		labelpad = 20)

	ax4.yaxis.set_label_coords(-0.15, 0.3)
	ax4.set_xticklabels([])

	plt.tick_params(axis = 'x', which = 'major', width = lwdth, length = 5)
	plt.tick_params(axis = 'y', which = 'major', width = lwdth, length = 0)

	major_yticks = np.linspace(0, max(StateMon.xpost[0])*1.1, 4)

	ax4.set_yticks(np.around(major_yticks, 1))

	plt.ylim(0, np.around(max(StateMon.xpost[0]), 1)*1.1)
	plt.xlim(0, t_run*1000)
	plt.xticks(size = s1)
	plt.yticks(size = s1)

# 5.6 ==== Rho state variable

ax5 = fig.add_subplot(gs[9, 0])
ax5.axhline(linestyle = 'dashed', color = 'dimgrey', lw = lwdth/2, 
	y = thr_b_rho, label = '$\\theta_{\\rho}$')

ax5.axhline(linestyle = 'solid', color = 'grey', lw = lwdth/2, 
	y = rho_max,
	label = 'UP state')
ax5.axhline(linestyle = 'dashed', color = 'black', lw = lwdth/2, 
	y = rho_min, 
	label = 'DOWN state')
ax5.plot(StateMon.t/ms, StateMon.rho[0], color = 'k', linewidth = lwdth)

plt.legend(loc = 'upper right', prop = {'size':s1-10}, 
	bbox_to_anchor = (1, 1.3),
	ncol = 3)
plt.ylabel('Efficacy $\\rho$ \n(a.u.)', size = s1, 
	horizontalalignment = 'center', 
	labelpad = 20)

ax5.set_xticklabels([])
ax5.yaxis.set_label_coords(-0.15, 0.3)

plt.tick_params(axis = 'x', which = 'major', width = lwdth, length = 5)
plt.tick_params(axis = 'y', which = 'major', width = lwdth, length = 0)

major_yticks = [rho_min, rho_max/2, rho_max]

ax5.set_yticks(major_yticks)

plt.ylim(-0.25, 1.25)
plt.xlim(0, t_run*1000)
plt.xticks(size = s1)
plt.yticks(size = s1)

# 5.7 ==== Weight

ax6 = fig.add_subplot(gs[11, 0])
ax6.plot(StateMon.t/ms, StateMon.w[0]/mV, label = 'w', color = 'k',
	linewidth = lwdth) 
ax6.axhline(color = 'grey', lw = lwdth, y = w_max/mV)

plt.ylabel('Weight $w$\n(mV)', size = s1, horizontalalignment = 'center',
	labelpad = 20)

ax6.yaxis.set_label_coords(-0.15, 0.3)

plt.tick_params(axis = 'x', which = 'major', width = lwdth, length = 5)
plt.tick_params(axis = 'y', which = 'major', width = lwdth, length = 0)

major_yticks = [0, w_max/mV/2, w_max/mV]

ax6.set_yticks(major_yticks) 

plt.ylim(-0.25, w_max/mV)
plt.xlim(0, t_run*1000)
plt.xticks(size = s1)
plt.yticks(size = s1)
plt.xlabel('Time (ms)', size = s1)

# 5.8 ==== Exporting plot to file

results_path = os.path.join(parent_dir, 'plots_results')
is_dir = os.path.isdir(results_path)

if not(is_dir):
	os.mkdir(results_path)

if exp_type == 'showcase':
	plot_name = sim_id + '_' + str(job_seed) + '_' + exp_type + '_' + plasticity_rule + '_' + parameter_set + str(bistability) + '.png'
else:
	plot_name = sim_id + '_' + str(job_seed) + '_' + exp_type + '_' + plasticity_rule + '_' + parameter_set + str(bistability) + '_' + str(pre_rate) + '_' + str(post_rate) + '.png'


plt.savefig(os.path.join(results_path, plot_name), 
	bbox_inches = 'tight', 
	dpi = 200)

# plt.show()

# END.

print('\nLR3_test.py - END.\n')