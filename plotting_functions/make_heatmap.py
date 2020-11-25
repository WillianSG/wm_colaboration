# -*- coding: utf-8 -*-
"""
@author: wgirao
@based-on: asonntag
"""
import os, pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors

# Target file
file = "_w_final_drho.pickle"

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Getting list of sim. results files
entries = os.listdir(os.path.join(parent_dir, 'sim_results'))

# Dir with .pickle data with results of simulations
sim_results_dir = os.path.join(parent_dir, 'sim_results', 
	entries[len(entries)-1], 
	entries[len(entries)-1]+file)

# Loading data
"""
[0] final_rho_all
[1] drho_all
[2] pre_freq
[3] post_freq
[4] min_freq
[5] max_freq
[6] step
[7] sim_id
[8] exp_type
[9] plasticity_rule
[10] parameter_set
[11] neuron_type
[12] bistability
[13] dt_resolution
[14] t_run
[15] noise
[16] N_Pre
[17] N_Post
[18] int_meth_syn
"""
exp_data = pickle.load(open(sim_results_dir, "rb" ))

# Results dir
results_path = os.path.join(parent_dir, 'plots_results')
is_dir = os.path.isdir(results_path)

if not(is_dir):
	os.mkdir(results_path)

plots_dir = os.path.join(results_path, entries[len(entries)-1])
is_dir = os.path.isdir(plots_dir)

if not(is_dir):
	os.mkdir(plots_dir)

# 1 ========== Matplotlib  ==========
lwdth = 1.0
tickfactor = 2
s1 = 37
s2 = 43

# ?
ticklabels = np.arange(exp_data[4], exp_data[5]+1, exp_data[6]*tickfactor)
yticklabels = np.arange(.1, 0.9+0.1, 0.1*tickfactor)

mpl.rcParams['axes.linewidth'] = lwdth
plt.close("all")

# Append all drho as a 1D array (for heat range)
drho_all_array = np.concatenate((exp_data[1]), axis = 0)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1) # "111" means "1x1 grid, first subplot"

# 1.1 ========== Rho (w) as func. of pre- and post- activity  ==========

plt.pcolor(exp_data[0].transpose(), cmap = 'Reds', vmin = 0, vmax = 1)

# setting-up heatmap grid
ax1.set_xticks(np.arange(0.5, len(exp_data[2]), tickfactor))
ax1.set_xticklabels(np.around(ticklabels), rotation = 40)
ax1.set_yticks(np.arange(0.5, len(exp_data[3]), tickfactor))
ax1.set_yticklabels(np.around(ticklabels))

plt.xlabel('Presynaptic $FR$ (Hz)')
plt.ylabel('Postsynaptic $FR$ (Hz)')

ax1.set_xlim(xmin=0, xmax=len(exp_data[2]))
ax1.set_ylim(ymin=0, ymax=len(exp_data[3]))

# setting-up heatmap colorbar
cb = plt.colorbar()
cb.set_label('Synaptic weight w (a.u)')

plt.savefig(os.path.join(plots_dir, 
	exp_data[7] + '_' + exp_data[8] + '_abs_w.png'),
	bbox_inches = 'tight', 
	dpi = 200)

# 1.2 ========== drho (final/init.) as func. of pre/post activity  ==========

# https://www.geeksforgeeks.org/matplotlib-colors-twoslopenorm-class-in-python/
norm = mcolors.TwoSlopeNorm(vmin = 0., vcenter = 1.0, vmax = 2.0)

fig = plt.figure(figsize=(21,21))
ax1 = fig.add_subplot(1, 1, 1)

plt.pcolor(exp_data[1].transpose(), 
	norm = norm, 
	vmin = min(drho_all_array), 
	vmax = max(drho_all_array),
	cmap = 'RdBu_r')

ticklabels = np.arange(exp_data[4], exp_data[5]+1, exp_data[6]*tickfactor)

ax1.set_xticks(np.arange(0.5, len(exp_data[2]), tickfactor))
ax1.set_xticklabels(np.around(ticklabels), rotation = 40)
ax1.set_yticks(np.arange(0.5, len(exp_data[3]), tickfactor))
ax1.set_yticklabels(np.around(ticklabels))
plt.xlabel('Presynaptic $FR$ (Hz)', size = s2, labelpad = 12)
plt.ylabel('Postsynaptic $FR$ (Hz)', size = s2, labelpad = 20)

ax1.set_xlim(xmin = 0, xmax = len(exp_data[2]))
ax1.set_ylim(ymin = 0, ymax = len(exp_data[3]))

cb = plt.colorbar()
cb.set_label(r'Change in synaptic efficacy $\Delta \rho$ (a.u.)',
	size = s2,
	labelpad = 20)

cb.ax.tick_params(width = lwdth,
	labelsize = s1,
	pad = 10, 
	direction = 'in')

plt.tick_params(axis = 'both',
	which = 'major',
	width = lwdth,
	length = 9,
	labelsize = s1,
	pad = 10, 
	direction = 'in')

plt.savefig(os.path.join(plots_dir, 
	exp_data[7] + '_' + exp_data[8] + "_drho.png"),
	bbox_inches = 'tight', 
	dpi = 200)

print('\nmake_heatmap.py - END.\n')