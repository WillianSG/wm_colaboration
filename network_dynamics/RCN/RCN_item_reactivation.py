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
import itertools
import os, sys, pickle, shutil
from brian2 import prefs, ms, Hz
import os.path as path
import numpy as np

import multiprocessing as mp

prefs.codegen.target = 'numpy'

helper_dir = 'helper_functions'
plotting_funcs_dir = 'plotting_functions'

# Parent directory
parent_dir = os.path.dirname(path.abspath(path.join(__file__, '../..')))

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))
sys.path.append(os.path.join(parent_dir, plotting_funcs_dir))

# Helper modules
from helper_functions.other import *
from helper_functions.recurrent_competitive_network import RecurrentCompetitiveNet
from plotting_functions.rcn_spiketrains_histograms import plot_rcn_spiketrains_histograms
from plotting_functions.plot_syn_matrix_heatmap import plot_syn_matrix_heatmap
from plotting_functions.plot_conn_matrix import plot_conn_matrix
from plotting_functions.plot import *
from plotting_functions.plot_video import generate_video
from plotting_functions.plot_x_u_spks_from_basin import plot_x_u_spks_from_basin

make_plots = True
plasticity_rule = 'LR4'
parameter_set = '2.2'
t_run = 1.0  # seconds
stim_pulse_duration = 20 * ms
percentage_stim_ids = 35  # percentage

plastic_syn = False
plastic_ux = True
stimulus_pulse = True
E_E_syn_matrix_snapshot = False

# 1 ------ initializing/running network ------
for p in np.arange(0, 100, 100):
    print('Percentage:', p, '%')

    rcn = RecurrentCompetitiveNet(
        plasticity_rule=plasticity_rule,
        parameter_set=parameter_set,
        t_run=t_run * second)

    rcn.stimulus_pulse = stimulus_pulse
    rcn.E_E_syn_matrix_snapshot = E_E_syn_matrix_snapshot
    rcn.w_e_i = 3 * mV  # for param. 2.1: 5*mV
    rcn.w_max = 10 * mV  # for param. 2.1: 10*mV
    rcn.spont_rate = 15 * Hz

    rcn.net_init()

    rcn.set_active_E_ids(stimulus='flat_to_E_fixed_size', offset=0)
    rcn.set_active_E_ids(stimulus='flat_to_E_fixed_size', offset=100)
    rcn.set_potentiated_synapses()

    rcn.send_PS(
        pattern='flat_to_E_fixed_size',
        frequency=rcn.stim_freq_e,
        stim_perc=35,
        offset=0)

    rcn.set_E_E_plastic(plastic=plastic_syn)
    rcn.set_E_E_ux_vars_plastic(plastic=plastic_ux)

    rcn.set_stimulus_pulse_duration(duration=stim_pulse_duration)

    rcn.run_net(duration=t_run, pulse_ending=stim_pulse_duration)

    rcn.run_net(duration=t_run, pulse_ending=stim_pulse_duration)

    # 2 ------ exporting simulation data ------

    rcn.pickle_E_E_syn_matrix_state()
    rcn.get_x_traces_from_pattern_neurons()
    rcn.get_u_traces_from_pattern_neurons()
    rcn.get_spks_from_pattern_neurons()

    # 3 ------ plotting simulation data ------

    plot_x_u_spks_from_basin(path=rcn.net_sim_data_path)
    plot_x_u_spks_from_basin(path=rcn.net_sim_data_path)

    plot_syn_matrix_heatmap(path_to_data=rcn.E_E_syn_matrix_path)

    plot_rcn_spiketrains_histograms(
        Einp_spks=rcn.get_Einp_spks()[0],
        Einp_ids=rcn.get_Einp_spks()[1],
        stim_E_size=rcn.stim_size_e,
        E_pop_size=rcn.N_input_e,
        Iinp_spks=rcn.get_Iinp_spks()[0],
        Iinp_ids=rcn.get_Iinp_spks()[1],
        stim_I_size=rcn.stim_size_i,
        I_pop_size=rcn.N_input_i,
        E_spks=rcn.get_E_spks()[0],
        E_ids=rcn.get_E_spks()[1],
        I_spks=rcn.get_I_spks()[0],
        I_ids=rcn.get_I_spks()[1],
        t_run=t_run,
        path_to_plot=rcn.net_sim_data_path)
