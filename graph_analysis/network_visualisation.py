# import
import os

import numpy as np

from brian2 import second, prefs, NeuronGroup, Synapses

from plotting_functions.rcn_spiketrains_histograms import plot_rcn_spiketrains_histograms

prefs.codegen.target = 'numpy'

# Helper modules
from helper_functions.recurrent_competitive_network import RecurrentCompetitiveNet
from plotting_functions.plot import *
from plotting_functions.graphing import *

clean_folder()

# 1 ------ initializing/running network ------

show_graphs = True
plasticity_rule = 'LR3'
parameter_set = '2.0'
# plasticity_rule = 'LR4'
# parameter_set = '2.0'

rcn = RecurrentCompetitiveNet(
        plasticity_rule=plasticity_rule,
        parameter_set=parameter_set,
        t_run=3 * second,
        save_path=os.getcwd() )

rcn.stimulus_pulse = True

rcn.net_init()

neurons_subsample = 0.05
edges_subsample = 1
# TODO plot without edges
#  ------ Create subsampled graph for visualisation and complete for analysis
g_initial = rcn2nx( rcn, neurons_subsample=neurons_subsample, subsample_attractors=True,
                    edges_subsample=edges_subsample,
                    output_filename='initial' )
g_initial_complete = rcn2nx( rcn, output_filename='initial_complete' )
nx2pyvis( g_initial, output_filename='initial', open_output=show_graphs )

# -------- First attractor
rcn.set_E_E_plastic( plastic=True )
rcn.set_stimulus_e( stimulus='flat_to_E_fixed_size', frequency=rcn.stim_freq_e, offset=0 )
rcn.set_stimulus_i( stimulus='flat_to_I', frequency=rcn.stim_freq_i )
rcn.run_net( period=2 )

g_first = rcn2nx( rcn, neurons_subsample=neurons_subsample, subsample_attractors=True,
                  edges_subsample=edges_subsample,
                  output_filename='first' )
g_first_complete = rcn2nx( rcn, output_filename='first_complete' )
nx2pyvis( g_first, output_filename='first', open_output=show_graphs )

comment = '---- First attractor learned'
print( comment )
print( 'Attractor inhibition',
       attractor_statistics( g_first_complete, 'inhibition',
                             comment=comment + f' (subsample=({neurons_subsample},{edges_subsample}))' ) )
print( 'Attractor excitation',
       attractor_statistics( g_first_complete, 'excitation',
                             comment=comment + f' (subsample=({neurons_subsample},{edges_subsample}))' ) )
print( 'Attractor self-excitation',
       attractor_statistics( g_first_complete, 'self-excitation',
                             comment=comment + f' (subsample=({neurons_subsample},{edges_subsample}))' ) )
print( 'Attractor connectivity',
       attractor_connectivity( g_first_complete,
                               comment=comment + f' (subsample=({neurons_subsample},{edges_subsample}))' ) )
print( 'Attractor algebraic connectivity',
       attractor_algebraic_connectivity( g_first_complete,
                                         comment=comment + f' (subsample=({neurons_subsample},{edges_subsample}))' ) )

# --------- Second attractor
rcn.stimulus_pulse_duration = 5 * second
rcn.set_stimulus_e( stimulus='flat_to_E_fixed_size', frequency=rcn.stim_freq_e, offset=100 )
rcn.set_stimulus_i( stimulus='flat_to_I', frequency=rcn.stim_freq_i )
rcn.run_net( period=2 )

g_second = rcn2nx( rcn, neurons_subsample=neurons_subsample, subsample_attractors=True,
                   edges_subsample=edges_subsample,
                   output_filename='second' )
g_second_complete = rcn2nx( rcn, output_filename='second_complete' )
nx2pyvis( g_second, output_filename='second', open_output=show_graphs )

comment = '---- Second attractor learned'
print( comment )
print( 'Attractor inhibition',
       attractor_statistics( g_second_complete, 'inhibition',
                             comment=comment + f' (subsample=({neurons_subsample},{edges_subsample}))' ) )
print( 'Attractor excitation',
       attractor_statistics( g_second_complete, 'excitation',
                             comment=comment + f' (subsample=({neurons_subsample},{edges_subsample}))' ) )
print( 'Attractor self-excitation',
       attractor_statistics( g_second_complete, 'self-excitation',
                             comment=comment + f' (subsample=({neurons_subsample},{edges_subsample}))' ) )
print( 'Attractor connectivity',
       attractor_connectivity( g_second_complete,
                               comment=comment + f' (subsample=({neurons_subsample},{edges_subsample}))' ) )
print( 'Attractor algebraic connectivity',
       attractor_algebraic_connectivity( g_second_complete,
                                         comment=comment + f' (subsample=({neurons_subsample},{edges_subsample}))' ) )

# TODO save plots and graphs in same RCN directory
# TODo evaluate connectivity and algebraic connectivity for inhibitory
plot_rcn_spiketrains_histograms(
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
        t_run=6,
        path_to_plot=os.getcwd(),
        show=True )

# save_graph_results()
