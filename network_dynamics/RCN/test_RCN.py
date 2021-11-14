# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System

Function:
-

Script arguments:
-

Script output:
-
"""
import setuptools
import os, sys, pickle, shutil
from brian2 import *
from numpy import *
from time import localtime
import os.path as path

import matplotlib as mpl
import matplotlib.pyplot as plt

prefs.codegen.target = 'numpy'

helper_dir = 'helper_functions'
plotting_funcs_dir = 'plotting_functions'

# Parent directory
parent_dir = os.path.dirname(path.abspath(path.join(__file__ , '../..')))

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))
sys.path.append(os.path.join(parent_dir, plotting_funcs_dir))

# Helper modules
from recurrent_competitive_network import RecurrentCompetitiveNet
from rcn_spiketrains_histograms import plot_rcn_spiketrains_histograms
from plot_syn_matrix_heatmap import plot_syn_matrix_heatmap
from plot_conn_matrix import plot_conn_matrix

# 1 ------ initializing/running network ------

rcn = RecurrentCompetitiveNet(
	plasticity_rule = 'LR3', 
	parameter_set = '2.0', 
	t_run = 3*second)

rcn.stimulus_pulse = True

rcn.net_init()

# rcn.set_random_E_E_syn_w(percent = 0.5)

rcn.stim_freq_e = 6600*Hz
rcn.stim_freq_i = 3900*Hz

rcn.set_stimulus_e(stimulus = 'circle', frequency = rcn.stim_freq_e)
rcn.set_stimulus_i(stimulus = 'random_I', frequency = rcn.stim_freq_i)

rcn.set_E_E_plastic(plastic = True)

rcn.run_net(period = 8)

rcn.get_target_spks(
	targets_E = [34, 60, 0, 4, 17, 25, 55, 103, 200, 39, 15], 
	targets_I = [0, 2, 15, 20, 35, 50, 60, 45, 29, 8, 26])

rcn.get_target_spks(all = True)

# 2 ------ plotting simulation data ------

plot_conn_matrix(
	conn_matrix = rcn.get_conn_matrix(pop = 'E_E'), 
	population = 'E_E', 
	path = rcn.get_sim_data_path())

plot_syn_matrix_heatmap(path_to_data = rcn.E_E_syn_matrix_path)

plot_rcn_spiketrains_histograms(
	Einp_spks = rcn.get_Einp_spks()[0],
	Einp_ids = rcn.get_Einp_spks()[1],
	stim_E_size = rcn.stim_size_e,
	E_pop_size = rcn.N_input_e,
	Iinp_spks = rcn.get_Iinp_spks()[0],
	Iinp_ids = rcn.get_Iinp_spks()[1],
	stim_I_size = rcn.stim_size_i,
	I_pop_size = rcn.N_input_i,
	E_spks = rcn.get_E_spks()[0],
	E_ids = rcn.get_E_spks()[1],
	I_spks = rcn.get_I_spks()[0],
	I_ids = rcn.get_I_spks()[1],
	t_run = rcn.t_run,
	path_to_plot = rcn.net_sim_data_path)