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
from brian2 import second, prefs
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
from other import *
from recurrent_competitive_network import RecurrentCompetitiveNet
from rcn_spiketrains_histograms import plot_rcn_spiketrains_histograms
from plot_syn_matrix_heatmap import plot_syn_matrix_heatmap
from plot_conn_matrix import plot_conn_matrix
from plot import *
from plot_video import generate_video

# 1 ------ initializing/running network ------

make_plots = True
plasticity_rule = 'LR4'
parameter_set = '2.0'

# TODO add pre/post to syn matrix axes
# TODO plot synaptic weights and compare with LR3 w*(x_*u)
rcn = RecurrentCompetitiveNet(
    plasticity_rule=plasticity_rule,
    parameter_set=parameter_set,
    t_run=3 * second)

rcn.stimulus_pulse = True

rcn.net_init()

rcn.set_E_E_plastic(plastic=True)

rcn.set_stimulus_e(stimulus='flat_to_E_fixed_size', frequency=rcn.stim_freq_e, offset=0)
rcn.set_stimulus_i(stimulus='flat_to_I', frequency=rcn.stim_freq_i)

rcn.run_net(duration=2)

"""
In line 550 on the net obj this variable is used to flag when the input stimulus has to terminate 
(basically it stops a second before the end of the simulation - line 58). 
Since the simulation is running for 6s (3s at each run), I`m updating it so that the 2nd stimulus input ends a second 
before the end of the 2nd simulation [this should be automatized].
"""
# TODO automatise stimulus pulse duration setting
rcn.stimulus_pulse_duration = 5 * second

rcn.set_stimulus_e(stimulus='flat_to_E_fixed_size', frequency=rcn.stim_freq_e, offset=100)
rcn.set_stimulus_i(stimulus='flat_to_I', frequency=rcn.stim_freq_i)

rcn.run_net(duration=2)

# TODO display connectivity somehow

# 2 ------ plotting simulation data ------
if make_plots:
    # TODO integrate windows with stimulus presentation settings
    windows = [(0, 3), (3, 6)] * second
    for w in windows:
        if rcn.plasticity_rule == 'LR4':
            plot_utilisation_resources(w, rcn.E_E, rcn.E_E_rec, rcn.E_mon, save_path=rcn.get_sim_data_path())
        plot_membrane_potentials(w, rcn.E_rec, rcn.E_mon, save_path=rcn.get_sim_data_path())
        plot_synaptic_weights(w, rcn.E_E, rcn.E_E_rec, rcn.E_mon, save_path=rcn.get_sim_data_path())

    population = "E_E"
    plot_conn_matrix(
        conn_matrix=rcn.get_conn_matrix(pop='E_E'),
        population=population,
        path=rcn.get_sim_data_path())

    plot_syn_matrix_heatmap(path_to_data=rcn.E_E_syn_matrix_path, show_last=False)

    if __name__ == "__main__":
        mp.set_start_method("fork")
        p = mp.Process(target=generate_video,
                       args=(rcn.get_sim_data_path(), population))
        p.start()

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
        t_run=6,
        path_to_plot=rcn.net_sim_data_path,
        show=False)
