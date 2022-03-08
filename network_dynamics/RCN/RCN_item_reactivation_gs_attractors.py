# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl / t.f.tiotto@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System / Artificial Intelligence

Function:
-

Script arguments:
-

Script output:
-
"""
import itertools
import os, sys, pickle, shutil
import os.path as path
import numpy as np
import argparse
import multiprocessing as mp

from brian2 import prefs, ms, Hz, mV
from helper_functions.recurrent_competitive_network import RecurrentCompetitiveNet
from helper_functions.other import *
from plotting_functions.x_u_spks_from_basin import plot_x_u_spks_from_basin
from plotting_functions.rcn_spiketrains_histograms import plot_rcn_spiketrains_histograms
from plotting_functions.spike_synchronisation import *

from plotting_functions import *

prefs.codegen.target = 'numpy'

# helper_dir = 'helper_functions'
# plotting_funcs_dir = 'plotting_functions'

# Parent directory
# parent_dir = os.path.dirname( path.abspath( path.join( __file__, '../..' ) ) )
#
# # Adding parent dir to list of dirs that the interpreter will search in
# sys.path.append( os.path.join( parent_dir, 'helper_functions' ) )
# sys.path.append( os.path.join( parent_dir, 'plotting_functions' ) )
#
# # Helper modules
# from helper_functions.other import *
# from helper_functions.recurrent_competitive_network import RecurrentCompetitiveNet
# from plotting_functions.rcn_spiketrains_histograms import plot_rcn_spiketrains_histograms
# from plotting_functions.plot_syn_matrix_heatmap import plot_syn_matrix_heatmap
# from plotting_functions.plot_conn_matrix import plot_conn_matrix
# from plotting_functions.plot import *
# from plotting_functions.plot_video import generate_video
# from plotting_functions.plot_x_u_spks_from_basin import plot_x_u_spks_from_basin

parser = argparse.ArgumentParser( description='RCN_item_reactivation_gs_attractors' )
parser.add_argument( '--ba', type=int, default=10, help='Background activity Hz' )
parser.add_argument( '--gs', type=int, default=20, help='Generic stimulus %' )
parser.add_argument( '--step', type=int, default=5, help='Step size for backgroun activity and generic stimulus' )
parser.add_argument( '--attractors', type=int, default=1, help='Number of attractors' )
parser.add_argument( '--show', type=str, default='False', help='Show output plots' )

args = parser.parse_args()

timestamp_folder = make_timestamped_folder( '../../results/RCN_attractor_reactivation/' )

plasticity_rule = 'LR4'
parameter_set = '2.2'

# TODO why does having 2+ attractors give better reactivation?
# TODO Mongillo mentions spike synchrony as important

# TODO when ba>0 a gs might not reactivate because a PS just happened.
#  Maybe fix x and u to a reasonable amount?
#  Or maybe run experiments multiple times and average?
#  Or maybe shorter experiments?

# TODO make this into one script for all different experiments?

# -- sweep over all combinations of parameters
print( f'Sweeping over all combinations of parameters: '
       f'background activity 0 Hz → {args.ba} Hz, generic stimulus 0 % → {args.gs} %, in steps of {args.step},' )

# background_activity = np.arange( 10, args.ba + args.step, args.step )
background_activity = [ 10 ]
# 1 ------ initializing/running network ------
i = 0
for ba in background_activity:
    # generic_stimulus = np.arange( 10, args.gs + args.step, args.step )
    generic_stimulus = [ 0 ]
    for gs in generic_stimulus:
        i += 1
        
        os.mkdir(
                os.path.join( timestamp_folder,
                              f'ba_{ba}_gs_{gs}' ) )
        save_dir = os.path.join( timestamp_folder,
                                 f'ba_{ba}_gs_{gs}' )
        
        # -- generic stimulus
        stim = (gs, (0.1, 0.2))
        
        print( f'Iteration {i} of {len( background_activity ) * len( generic_stimulus )}', end=' : ' )
        print( 'ba = ', ba, 'Hz', end=' , ' )
        print( 'gs = ', gs, '%' )
        
        rcn = RecurrentCompetitiveNet(
                plasticity_rule=plasticity_rule,
                parameter_set=parameter_set )
        
        plastic_syn = False
        plastic_ux = True
        rcn.E_E_syn_matrix_snapshot = False
        rcn.w_e_i = 3 * mV  # for param. 2.1: 5*mV
        rcn.w_max = 10 * mV  # for param. 2.1: 10*mV
        rcn.spont_rate = ba * Hz
        
        rcn.net_init()
        rcn.net_sim_data_path = save_dir
        
        attractors = [ ]
        if args.attractors >= 1:
            stim1_ids = rcn.set_active_E_ids( stimulus='flat_to_E_fixed_size', offset=0 )
            rcn.set_potentiated_synapses( stim1_ids )
            A1 = list( range( 0, 64 ) )
            attractors.append( ('A1', A1) )
        if args.attractors >= 2:
            stim2_ids = rcn.set_active_E_ids( stimulus='flat_to_E_fixed_size', offset=100 )
            rcn.set_potentiated_synapses( stim2_ids )
            A2 = list( range( 100, 164 ) )
            attractors.append( ('A2', A2) )
        if args.attractors >= 3:
            stim3_ids = rcn.set_active_E_ids( stimulus='flat_to_E_fixed_size', offset=180 )
            rcn.set_potentiated_synapses( stim3_ids )
            A3 = list( range( 180, 244 ) )
            attractors.append( ('A3', A3) )
        
        rcn.set_E_E_plastic( plastic=plastic_syn )
        rcn.set_E_E_ux_vars_plastic( plastic=plastic_ux )
        
        # run network, give generic pulse, run network again
        rcn.run_net( duration=stim[ 1 ][ 0 ] )
        act_ids = rcn.generic_stimulus( frequency=rcn.stim_freq_e, stim_perc=stim[ 0 ], subset=stim1_ids )
        rcn.run_net( duration=stim[ 1 ][ 0 ] + (stim[ 1 ][ 1 ] - stim[ 1 ][ 0 ]) )
        rcn.generic_stimulus_off( act_ids )
        rcn.run_net( duration=10 )
        
        # 2 ------ exporting simulation data ------
        
        rcn.pickle_E_E_syn_matrix_state()
        rcn.get_x_traces_from_pattern_neurons()
        rcn.get_u_traces_from_pattern_neurons()
        rcn.get_spks_from_pattern_neurons()
        rcn.get_spikes_pyspike()
        
        # 3 ------ plotting simulation data ------
        
        fig1 = plot_x_u_spks_from_basin( path=save_dir, filename=f'x_u_spks_from_basin_ba_{ba}_gs_{gs}',
                                         title_addition=f'background activity {ba} Hz, generic stimulus {gs} %',
                                         generic_stimulus=stim,
                                         attractors=attractors,
                                         show=args.show )
        
        # plot_syn_matrix_heatmap( path_to_data=rcn.E_E_syn_matrix_path )
        
        # fig2 = plot_rcn_spiketrains_histograms(
        #         Einp_spks=rcn.get_Einp_spks()[ 0 ],
        #         Einp_ids=rcn.get_Einp_spks()[ 1 ],
        #         stim_E_size=rcn.stim_size_e,
        #         E_pop_size=rcn.N_input_e,
        #         Iinp_spks=rcn.get_Iinp_spks()[ 0 ],
        #         Iinp_ids=rcn.get_Iinp_spks()[ 1 ],
        #         stim_I_size=rcn.stim_size_i,
        #         I_pop_size=rcn.N_input_i,
        #         E_spks=rcn.get_E_spks()[ 0 ],
        #         E_ids=rcn.get_E_spks()[ 1 ],
        #         I_spks=rcn.get_I_spks()[ 0 ],
        #         I_ids=rcn.get_I_spks()[ 1 ],
        #         t_run=rcn.net.t,
        #         path=save_dir,
        #         filename=f'rcn_population_spiking_ba_{ba}_gs_{gs}',
        #         title_addition=f'background activity {ba} Hz, generic stimulus {gs} %',
        #         show=args.show )
