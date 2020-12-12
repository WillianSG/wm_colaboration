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
from time import *
import multiprocessing
# from pyspike import *
prefs.codegen.target = 'numpy'

#=====
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors
#=====

helper_dir = 'helper_functions'

sys.path.append('PySpike')
from pyspike import SpikeTrain
import pyspike as spk

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

# Helper modules
from single_poisson_spk_gen import *
from poisson_spiking_gen import *

# 1 ========== Execution parameters ==========

dt_resolution = 0.0001
t_run = 5 # simulation time (seconds)
step = 5

noise = 0.25

min_freq = 0
max_freq = 100

repetition = 1 # number of times exp repeated

isi_correlation = "positive"

# Frequency activity ranges (for pre and post neurons)
pre_freq = arange(min_freq, max_freq+0.1, step)
post_freq = arange(min_freq, max_freq+0.1, step)

# ?
spike_diff = zeros((len(pre_freq), len(post_freq), repetition))
spike_diff_corr = zeros((len(pre_freq), len(post_freq), repetition))

spike_diff_corr_all_concat_rep = zeros((len(pre_freq)*len(post_freq),
	repetition))

spike_diff_all_concat_rep = zeros((len(pre_freq)*len(post_freq), 
	repetition))

# IMPORTANT - have to improve this portion since we're using jobseed 
for n in arange(0, repetition, 1):
	print('Repetition #: ', n)
	for p in arange(0, len(pre_freq), 1):
		for q in arange(0, len(post_freq), 1):
			if p > 0 or q > 0:
				# independent Poisson spikes
				ind_train1 = single_poisson_spk_gen(pre_freq[p], t_run, dt_resolution, job_seed = job_seed)
				ind_train2 = single_poisson_spk_gen(post_freq[q], t_run, dt_resolution, job_seed = job_seed+n+1)

				# PySpike
				"""
				SpikeTrain - obj consists of the spike times given as numpy arrays as well as the edges of the spike train as [t_start, t_end].
				"""
				ind_Train1 = SpikeTrain(ind_train1.flatten(), [0.0, t_run])
				ind_Train2 = SpikeTrain(ind_train2.flatten(), [0.0, t_run])

				# corr_train1, corr_train2 = poisson_generation_rate(pre_freq[p],post_freq[q], t_run, dt, noise = .25)

				# ?
				corr_train1, corr_train2 = poisson_spiking_gen(
					rate_pre = pre_freq[p], 
					rate_post = post_freq[q], 
					t_run = t_run, 
					dt = dt_resolution, 
					noise = noise,
					job_seed = job_seed+n+2,
					correlation = isi_correlation)

				# PySpike object
				Corr_train1 = SpikeTrain(corr_train1.flatten(), 
					edges = (0.0, t_run))
				Corr_train2 = SpikeTrain(corr_train2.flatten(), 
					edges = (0.0, t_run))

				spike_profile_corr = spk.spike_profile(Corr_train1, Corr_train2)

				spike_diff_corr[p, q, n] = spike_profile_corr.avrg()

				spike_profile = spk.spike_profile(ind_Train1, ind_Train2)

				spike_diff[p, q, n] = spike_profile.avrg()

	spike_diff_corr_all_concat = concatenate((spike_diff_corr[:,:,n]), 
		axis = 0)

	spike_diff_all_concat = concatenate((spike_diff[:,:,n]), axis = 0)

	spike_diff_corr_all_concat_rep[:,n] = spike_diff_corr_all_concat
	spike_diff_all_concat_rep [:,n] = spike_diff_all_concat

spike_diff_corr_all_concat_rep_mean = spike_diff_corr_all_concat_rep.mean((1))
spike_diff_all_concat_rep_mean = spike_diff_all_concat_rep.mean((1))
spike_diff_mean = spike_diff.mean((2))
spike_diff_corr_mean = spike_diff_corr.mean((2))

# 2 Plotting =======================================

tickfactor=2
s1 = 37
s2  = 43
lwdth = 3
mpl.rcParams['axes.linewidth'] = lwdth #set the value globally

# class from Stackoverflow:
# https://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib (11.01.2018)
class MidpointNormalize(Normalize):
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y))

norm = MidpointNormalize(midpoint=0.25)

# drho as function of pre- and postsynaptic activity
fig = plt.figure(figsize=(21,21))
ax1 = fig.add_subplot(1,1,1)

plt.pcolor(spike_diff_corr_mean.transpose(), norm = norm, 
	vmin = min(spike_diff_all_concat_rep_mean),
	vmax = max(spike_diff_all_concat_rep_mean),
	cmap = 'Blues')

ticklabels = np.arange(min_freq, max_freq+1, step*tickfactor)

ax1.set_xticks(np.arange(0.5,len(pre_freq),tickfactor))
ax1.set_xticklabels(np.around(ticklabels), rotation = 40)

ax1.set_yticks(np.arange(0.5,len(post_freq),tickfactor))
ax1.set_yticklabels(np.around(ticklabels))

plt.xlabel('Presynaptic $FR$ (Hz)',size=s2,labelpad=12)
plt.ylabel('Postsynaptic $FR$ (Hz)',size=s2,labelpad=20)

ax1.set_xlim(xmin=0, xmax=len(pre_freq))
ax1.set_ylim(ymin=0, ymax=len(post_freq))

cb = plt.colorbar()
cb.set_label('SPIKE Distance',size=s2,labelpad=20) #correlated
cb.ax.tick_params(width=lwdth,labelsize=s1,pad=10, direction = 'in')

plt.tick_params(axis='both',which='major',width=lwdth, length=9, 
	labelsize=s1,
	pad=10, 
	direction = 'in')

plt.savefig(isi_correlation+'Correlated_Spike_Distance_10Repetition.png',
	bbox_inches='tight', 
	dpi = 200)


fig = plt.figure(figsize=(21,21))
ax1=fig.add_subplot(1,1,1)

plt.pcolor(spike_diff_mean.transpose(), norm = norm, 
	vmin = min(spike_diff_all_concat_rep_mean),
	vmax = max(spike_diff_all_concat_rep_mean),
	cmap = 'Blues')

ticklabels = np.arange(min_freq, max_freq+1, step*tickfactor)

ax1.set_xticks(np.arange(0.5,len(pre_freq),tickfactor))
ax1.set_xticklabels(np.around(ticklabels), rotation = 40)
ax1.set_yticks(np.arange(0.5,len(post_freq),tickfactor))
ax1.set_yticklabels(np.around(ticklabels))

plt.xlabel('Presynaptic $FR$ (Hz)',size=s2,labelpad=12)
plt.ylabel('Postsynaptic $FR$ (Hz)',size=s2,labelpad=20)

ax1.set_xlim(xmin=0, xmax=len(pre_freq))
ax1.set_ylim(ymin=0, ymax=len(post_freq))

cb = plt.colorbar()
cb.set_label('SPIKE Distance',size=s2,labelpad=20)#independent
cb.ax.tick_params(width=lwdth,labelsize=s1,pad=10, direction = 'in')

plt.tick_params(axis='both',which='major',width=lwdth,
	length=9,
	labelsize=s1,
	pad=10, 
	direction = 'in')

plt.savefig('Independent_Spike_Distance_10Repetition.png',
	bbox_inches='tight', 
	dpi = 200)

print('\nrun_correla_spk_dist.py - END.\n')