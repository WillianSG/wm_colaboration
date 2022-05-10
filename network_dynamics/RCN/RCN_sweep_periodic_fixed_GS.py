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
import warnings
from tqdm import TqdmWarning
import timeit
from itertools import product

from brian2 import prefs, ms, Hz, mV
from helper_functions.recurrent_competitive_network import RecurrentCompetitiveNet
from helper_functions.other import *
from plotting_functions.x_u_spks_from_basin import plot_x_u_spks_from_basin
from plotting_functions.rcn_spiketrains_histograms import plot_rcn_spiketrains_histograms
from plotting_functions.spike_synchronisation import *

from plotting_functions import *

prefs.codegen.target = 'numpy'

np.warnings.filterwarnings( 'ignore', category=np.VisibleDeprecationWarning )
warnings.filterwarnings( 'ignore', category=TqdmWarning )

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
parser.add_argument( '--ba_amount', type=int, default=[ 10, 10 ], nargs='+',
                     help='Bounds for background activity in Hz' )
parser.add_argument( '--ba_step', type=int, default=5, help='Step size for background activity' )
parser.add_argument( '--gs_amount', type=int, default=[ 15, 15 ], nargs='+',
                     help='Bounds for generic stimulus in % of stimulated neurons' )
parser.add_argument( '--gs_step', type=int, default=5, help='Step size for generic stimulus' )
parser.add_argument( '--i_amount', type=int, default=[ 15, 15 ], nargs='+',
                     help='Bounds for I-to-E inhibition weight' )
parser.add_argument( '--i_step', type=int, default=2, help='Step size for I-to-E inhibition weight' )
parser.add_argument( '--i_stim_amount', type=int, default=[ 15, 15 ], nargs='+',
                     help='Bounds for I inhibition input in Hz' )
parser.add_argument( '--i_stim_step', type=int, default=10, help='Step size for I inhibition' )
parser.add_argument( '--gs_freq', type=float, default=4, help='Frequency of generic stimulus Hz' )
parser.add_argument( '--gs_length', type=float, default=0.1, help='Length of generic stimulus in seconds' )
parser.add_argument( '--pre_runtime', type=float, default=0.0, help='Runtime before showing generic stimulus' )
parser.add_argument( '--gs_runtime', type=float, default=15, help='Runtime for showing generic stimulus' )
parser.add_argument( '--post_runtime', type=float, default=0.0, help='Runtime after showing generic stimulus' )
parser.add_argument( '--attractors', type=int, default=2, choices=[ 1, 2, 3 ], help='Number of attractors' )
parser.add_argument( '--show', default=False, action=argparse.BooleanOptionalAction )

args = parser.parse_args()

if len( args.ba_amount ) == 1:
    args.ba_amount = [ args.ba_amount[ 0 ], args.ba_amount[ 0 ] ]
elif len( args.ba_amount ) == 2:
    pass
else:
    raise ValueError( 'Wrong number of background activity amounts' )
if len( args.gs_amount ) == 1:
    args.gs_amount = [ args.gs_amount[ 0 ], args.gs_amount[ 0 ] ]
elif len( args.gs_amount ) == 2:
    pass
else:
    raise ValueError( 'Wrong number of generic stimulus amounts' )
if len( args.i_amount ) == 1:
    args.i_amount = [ args.i_amount[ 0 ], args.i_amount[ 0 ] ]
elif len( args.i_amount ) == 2:
    pass
else:
    raise ValueError( 'Wrong number of I-to-E weight amounts' )
if len( args.i_stim_amount ) == 1:
    args.i_stim_amount = [ args.i_stim_amount[ 0 ], args.i_stim_amount[ 0 ] ]
elif len( args.i_stim_amount ) == 2:
    pass
else:
    raise ValueError( 'Wrong number of I input frequency amounts' )

assert args.ba_amount[ 0 ] <= args.ba_amount[
    1 ], 'Lower bound for background activity Hz must be smaller than upper bound'
assert args.gs_amount[ 0 ] <= args.gs_amount[ 1 ], 'Lower bound for generic stimulus % must be smaller than upper bound'
assert args.gs_freq > 0, 'Generic stimulus frequency must be positive'
assert args.ba_step > 0, 'Step size must be positive'
assert args.gs_step > 0, 'Step size must be positive'
assert args.i_step > 0, 'Step size must be positive'
assert args.i_stim_step > 0, 'Step size must be positive'

timestamp_folder = make_timestamped_folder( '../../results/RCN_engineering_reactivation/' )

plasticity_rule = 'LR4'
parameter_set = '2.2'
"""
tau_d = 90 * ms  # x's
tau_f = 700 * ms  # u's'
"""
# -- sweep over all combinations of parameters
background_activity = np.arange( args.ba_amount[ 0 ], args.ba_amount[ 1 ] + args.ba_step, args.ba_step )
generic_stimuli = np.arange( args.gs_amount[ 0 ], args.gs_amount[ 1 ] + args.gs_step, args.gs_step )
I_E_weights = np.arange( args.i_amount[ 0 ], args.i_amount[ 1 ] + args.i_step, args.i_step )
I_input_freq = np.arange( args.i_stim_amount[ 0 ], args.i_stim_amount[ 1 ] + args.i_stim_step, args.i_stim_step )
parameter_combinations = itertools.product( background_activity, generic_stimuli, I_E_weights, I_input_freq )
# 1 ------ initializing/running network ------
i = 0
running_times = [ ]
for ba, gs_percentage, i_e_w, i_freq in parameter_combinations:
    # --initialise time to predict overall experiment time
    start_time = timeit.default_timer()
    i += 1
    
    rcn = RecurrentCompetitiveNet(
            plasticity_rule=plasticity_rule,
            parameter_set=parameter_set )
    
    plastic_syn = False
    plastic_ux = True
    rcn.E_E_syn_matrix_snapshot = False
    rcn.w_e_i = 3 * mV  # for param. 2.1: 5*mV
    rcn.w_max = 10 * mV  # for param. 2.1: 10*mV
    rcn.spont_rate = ba * Hz
    
    rcn.w_e_i = 3 * mV  # 3 mV default
    rcn.w_i_e = i_e_w * mV  # 1 mV default
    
    rcn.U = 0.2  # 0.2 default
    
    rcn.net_init()
    
    rcn.set_stimulus_i( stimulus='flat_to_I', frequency=i_freq * Hz )  # default: 15 Hz
    
    save_dir = os.path.join( os.path.join( timestamp_folder, f'BA_{ba}_GS_{gs_percentage}_W_{i_e_w}_I_{i_freq}' ) )
    os.mkdir( save_dir )
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
    
    # -- generic stimuli --
    gss_periodic = generate_periodic_gss( gs_percentage, args.gs_freq, args.gs_length,
                                          args.pre_runtime,
                                          args.gs_runtime,
                                          target=len( rcn.E ) )
    # gss_A1 = generate_gss( 60, args.gs_length, args.pre_runtime, args.gs_runtime,
    #                        target=stim1_ids, length= )
    gss_A1 = (100, stim1_ids, (args.pre_runtime, args.pre_runtime + args.gs_length),
              abs( args.pre_runtime + args.gs_length - gss_periodic[ 0 ][ 2 ][ 0 ] ))
    gss_periodic.insert( 0, gss_A1 )
    gss = gss_periodic
    # gss[ 0 ][ 3 ] = abs( gss[ 0 ][ 2 ][ 1 ] - gss[ 1 ][ 2 ][ 0 ] )
    # gss = compile_overlapping_gss( gss_periodic, gss_A1 )
    
    rcn.set_E_E_plastic( plastic=plastic_syn )
    rcn.set_E_E_ux_vars_plastic( plastic=plastic_ux )
    
    print( '-------------------------------------------------------' )
    print(
            f'Iteration {i} of {len( list( itertools.product( background_activity, generic_stimuli, I_E_weights, I_input_freq ) ) )}',
            end=' : ' )
    print( 'ba = ', ba, 'Hz', end=' , ' )
    print( 'gs = ', gs_percentage, '%', end=' , ' )
    print( 'I input = ', i_freq, 'Hz', 'I-to-E weight', i_e_w, 'mV' )
    
    # run network, give generic pulses, run network again
    stimulation_amount = [ ]
    rcn.run_net( duration=args.pre_runtime )
    for gs in gss:
        act_ids = rcn.generic_stimulus( frequency=rcn.stim_freq_e, stim_perc=gs[ 0 ], subset=gs[ 1 ] )
        stimulation_amount.append(
                (100 * np.intersect1d( act_ids, stim1_ids ).size / len( stim1_ids ),
                 100 * np.intersect1d( act_ids, stim2_ids ).size / len( stim2_ids ))
                )
        rcn.run_net( duration=gs[ 2 ][ 1 ] - gs[ 2 ][ 0 ] )
        rcn.generic_stimulus_off( act_ids )
        rcn.run_net( duration=gs[ 3 ] )
    
    rcn.run_net( duration=args.post_runtime )
    
    # 2 ------ exporting simulation data ------
    
    rcn.pickle_E_E_syn_matrix_state()
    rcn.get_x_traces_from_pattern_neurons()
    rcn.get_u_traces_from_pattern_neurons()
    rcn.get_spks_from_pattern_neurons()
    rcn.get_spikes_pyspike()
    
    # -- save the PS statistics for this iteration
    for atr in attractors:
        find_ps( save_dir, rcn.net.t, atr, write_to_file=True,
                 parameters={ 'ba_Hz'           : ba,
                              'gs_%'            : gs_percentage,
                              'I_to_E_weight_mV': i_e_w,
                              'I_input_freq_Hz' : i_freq
                              } )
    # count_pss_in_gss( save_dir, num_gss=len( gss ), write_to_file=True, ba=ba, gss=gss )
    
    # 3 ------ plotting simulation data ------
    title_addition = f'BA {ba} Hz, GS {gs_percentage} %, I-to-E {i_e_w} mV, I input {i_freq} Hz'
    filename_addition = f'_BA_{ba}_GS_{gs_percentage}_W_{i_e_w}_Hz_{i_freq}'
    
    fig1 = plot_x_u_spks_from_basin( path=save_dir,
                                     filename='x_u_spks_from_basin' + filename_addition,
                                     title_addition=title_addition,
                                     generic_stimuli=gss,
                                     attractors=attractors,
                                     num_neurons=len( rcn.E ),
                                     show=args.show )
    
    # plot_syn_matrix_heatmap( path_to_data=rcn.E_E_syn_matrix_path )
    
    fig2 = plot_rcn_spiketrains_histograms(
            Einp_spks=rcn.get_Einp_spks()[ 0 ],
            Einp_ids=rcn.get_Einp_spks()[ 1 ],
            stim_E_size=rcn.stim_size_e,
            E_pop_size=rcn.N_input_e,
            Iinp_spks=rcn.get_Iinp_spks()[ 0 ],
            Iinp_ids=rcn.get_Iinp_spks()[ 1 ],
            stim_I_size=rcn.stim_size_i,
            I_pop_size=rcn.N_input_i,
            E_spks=rcn.get_E_spks()[ 0 ],
            E_ids=rcn.get_E_spks()[ 1 ],
            I_spks=rcn.get_I_spks()[ 0 ],
            I_ids=rcn.get_I_spks()[ 1 ],
            t_run=rcn.net.t,
            path=save_dir,
            filename='rcn_population_spiking' + filename_addition,
            title_addition=title_addition,
            show=args.show )
    
    # 4 ------ saving PS statistics ------
    # -- append PS statistics for this iteration into one file for the whole experiment
    append_pss_to_xlsx( timestamp_folder, save_dir )
    # -- delete .pickle files as they're just too large to store
    remove_pickles( timestamp_folder )
    
    running_times.append( timeit.default_timer() - start_time )
    print(
            f'Predicted time remaining in experiment: '
            f'{np.round( np.mean( running_times ) * len( list( itertools.product( background_activity, generic_stimuli, I_E_weights, I_input_freq ) ) ) ) - i} s' )
    
    print( 'DEBUG\n% neurons stimulated by GS in each attractor\n', stimulation_amount, '\n' )
    
    # 5 ------ compute PS statistics for the whole experiment and write back to excel ------
    df_statistics = compute_pss_statistics( timestamp_folder,
                                            parameters=[ 'ba_Hz', 'gs_%', 'I_to_E_weight_mV', 'I_input_freq_Hz' ] )
    # TODO am I just printing the attractor neurons in raster plot?