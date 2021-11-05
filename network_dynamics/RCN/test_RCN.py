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

rcn = RecurrentCompetitiveNet(
	plasticity_rule = 'LR2', 
	parameter_set = '2.4', 
	t_run = 2*second)

rcn.net_init()
rcn.set_stimulus_e(stimulus = 'cross', frequency = rcn.stim_freq_e)
rcn.set_stimulus_i(stimulus = 'flat_to_I', frequency = rcn.stim_freq_i)

rcn.run_net(period = 4)

E_mon_pop = rcn.get_E_spks()
I_mon_pop = rcn.get_I_spks()
Einp_mon_pop = rcn.get_Einp_spks()
Iinp_mon_pop = rcn.get_Iinp_spks()

plot_rcn_spiketrains_histograms(
	Einp_spks = Einp_mon_pop[0],
	Einp_ids = Einp_mon_pop[1],
	stim_E_size = rcn.stim_size_e,
	Iinp_spks = Iinp_mon_pop[0],
	Iinp_ids = Iinp_mon_pop[1],
	stim_I_size = rcn.stim_size_i,
	E_spks = E_mon_pop[0],
	E_ids = E_mon_pop[1],
	I_spks = I_mon_pop[0],
	I_ids = I_mon_pop[1],
	t_run = rcn.t_run,
	path_to_plot = rcn.net_sim_data_path)