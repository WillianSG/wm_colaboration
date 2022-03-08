# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System

Function:
- Plots synaptic traces recorded during simulation only for pair of neurons that are part of the input stimulus.

Script arguments:
-

Script output:
-
"""
import os, sys, pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from brian2 import second, ms, units
import numpy as np
from helper_functions.other import *


def plot_x_u_spks_from_basin( path, generic_stimulus=None, attractors=None,
                              filename=None, title_addition='', show=True ):
    plt.close( 'all' )
    
    axis_label_size = 8
    font_size1 = 10
    font_size2 = 8
    
    linewidth1 = 0.05
    linewidth2 = 1.0
    linewidth3 = 1.5
    linewidth4 = 2.5
    
    label_size1 = 12
    label_size2 = 10
    
    alpha1 = 0.1
    alpha2 = 0.3
    alpha3 = 0.7
    
    u_color = 'b'
    x_color = 'r'
    ux_color = 'purple'
    
    us_pickled_data = os.path.join(
            path,
            'us_neurs_with_input.pickle' )
    
    with open( us_pickled_data, 'rb' ) as f:
        (
                us_neurs_with_input,
                sim_t_array,
                U,
                tau_f,
                # stim_pulse_duration
                ) = pickle.load( f )
    
    xs_pickled_data = os.path.join(
            path,
            'xs_neurs_with_input.pickle' )
    
    with open( xs_pickled_data, 'rb' ) as f:
        (
                xs_neurs_with_input,
                sim_t_array,
                tau_d,
                # stim_pulse_duration
                ) = pickle.load( f )
    
    spks_pickled_data = os.path.join(
            path,
            'spks_neurs_with_input.pickle' )
    
    with open( spks_pickled_data, 'rb' ) as f:
        (
                spk_mon_ids,
                spk_mon_ts,
                t_run) = pickle.load( f )
    
    xs = [ ]
    us = [ ]
    for x in range( 0, len( xs_neurs_with_input ) ):
        xs.append( xs_neurs_with_input[ x ][ 'x' ] )
        us.append( us_neurs_with_input[ x ][ 'u' ] )
    
    x_times_u = (np.mean( xs, axis=0 ) * np.mean( us, axis=0 )) / U
    
    x_mean = np.mean( xs, axis=0 )
    x_std = np.std( xs, axis=0 )
    
    u_mean = np.mean( us, axis=0 )
    u_std = np.std( us, axis=0 )
    
    fig = plt.figure( constrained_layout=True, figsize=(10, 4) )
    
    widths = [ 10 ]
    heights = [ 2, 2, 2 ]
    
    spec2 = gridspec.GridSpec(
            ncols=len( widths ),
            nrows=len( heights ),
            width_ratios=widths,
            height_ratios=heights,
            figure=fig )
    
    f0_ax1 = fig.add_subplot( spec2[ 0, 0 ] )
    
    # 1st y axis: x's
    # for x in range(0, len(xs_neurs_with_input)):
    # 	f0_ax1.plot(
    # 		sim_t_array,
    # 		xs_neurs_with_input[x]['x'],
    # 		color = x_color,
    # 		zorder = 0,
    # 		linewidth = linewidth1,
    # 		alpha = alpha1)
    
    f0_ax1.plot(
            sim_t_array,
            x_mean,
            color=x_color,
            zorder=0,
            linewidth=linewidth2 )
    
    f0_ax1.fill_between(
            sim_t_array,
            x_mean + x_std,
            x_mean - x_std,
            color=x_color,
            alpha=alpha2 )
    
    f0_ax1.set_ylabel(
            'x (a.u.)',
            size=axis_label_size,
            color=x_color )
    
    f0_ax1.tick_params( axis='y', labelcolor=x_color )
    
    f0_ax1.set_ylim( 0, 1 )
    f0_ax1.set_xlim( 0, sim_t_array[ -1 ] )
    
    # 2nd y axis: u's
    f0_ax2 = f0_ax1.twinx()
    
    # for x in range(0, len(us_neurs_with_input)):
    # 	f0_ax2.plot(
    # 		sim_t_array,
    # 		us_neurs_with_input[x]['u'],
    # 		zorder = 0,
    # 		linewidth = linewidth1,
    # 		color = u_color,
    # 		alpha = alpha1)
    
    f0_ax2.plot(
            sim_t_array,
            u_mean,
            zorder=0,
            linewidth=linewidth2,
            color=u_color )
    
    f0_ax2.fill_between(
            sim_t_array,
            u_mean + u_std,
            u_mean - u_std,
            color=u_color,
            alpha=alpha2 )
    
    f0_ax2.set_ylabel(
            'u (a.u.)',
            size=axis_label_size,
            color=u_color )
    
    f0_ax2.tick_params( axis='y', labelcolor=u_color )
    
    f0_ax2.set_ylim( 0, 1 )
    f0_ax2.set_xlim( 0, sim_t_array[ -1 ] )
    
    # plt.axvspan(
    #         0.0,
    #         (stim_pulse_duration / ms) / 1000,
    #         facecolor='grey',
    #         alpha=alpha2,
    #         label='PS (' + str( stim_pulse_duration ) + ')' )
    
    # plt.legend( loc='upper right', fontsize=font_size2 )
    
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels( [ ] )
    
    # plt.xticks( np.arange(
    #         0.0,
    #         sim_t_array[ -1 ] + 0.5,
    #         step=0.5 ) )
    
    # spks
    f1_ax1 = fig.add_subplot( spec2[ 1, 0 ] )
    
    # -- plot neuronal spikes with attractors in different colours
    if attractors:
        for atr in attractors:
            spk_mon_ts = np.array( spk_mon_ts )
            spk_mon_ids = np.array( spk_mon_ids )
            
            color = next( f1_ax1._get_lines.prop_cycler )[ 'color' ]
            
            atr_indexes = np.argwhere(
                    np.logical_and( np.array( spk_mon_ids ) >= atr[ 1 ][ 0 ],
                                    np.array( spk_mon_ids ) <= atr[ 1 ][ -1 ] ) )
            atr_spks = spk_mon_ts[ atr_indexes ]
            atr_ids = spk_mon_ids[ atr_indexes ]
            
            f1_ax1.plot( atr_spks, atr_ids, '|', color=color, zorder=0 )
    else:
        f1_ax1.plot( spk_mon_ts, spk_mon_ids, '|', color='black', zorder=0 )
    
    # f0_ax1.set_ylim( 0, n_neurons )
    f1_ax1.set_xlim( 0, sim_t_array[ -1 ] )
    
    # plt.xticks( np.arange(
    #         0.0,
    #         sim_t_array[ -1 ] + 0.5,
    #         step=0.5 ) )
    
    f1_ax1.set_ylabel(
            'Neuron ID',
            size=axis_label_size,
            color='k' )
    
    f1_ax2 = f1_ax1.twinx()
    
    f1_ax2.plot(
            sim_t_array,
            x_times_u,
            zorder=5,
            linewidth=linewidth4,
            color=ux_color,
            alpha=alpha3 )
    
    f1_ax2.set_ylabel(
            'x*u*1/U \n (a.u.)',
            size=axis_label_size,
            color=ux_color )
    
    plt.yticks( np.arange(
            0.0,
            6.0,
            step=1.0 ) )
    
    f1_ax2.tick_params( axis='y', labelcolor=ux_color )
    
    f2_ax1 = fig.add_subplot( spec2[ 2, 0 ] )
    
    # -- plot spike sync profile
    if attractors:
        for atr in attractors:
            x, y, y_smooth, pss = find_ps( path, sim_t_array[ -1 ], atr[ 1 ] )
            
            color = next( f2_ax1._get_lines.prop_cycler )[ 'color' ]
            
            f2_ax1.plot( x, y, color=color, alpha=0.5, label=atr[ 0 ] )
            f2_ax1.plot( x, y_smooth, '.', markersize=0.5, color=color )
            
            for ps in pss:
                print( f'Found PS in {atr[ 0 ]} '
                       f'between {x[ ps[ 0 ] ]} s and {x[ ps[ 1 ] ]} s '
                       f'centered at {x[ ps[ 0 ] + np.argmax( y_smooth[ ps[ 0 ]:ps[ 1 ] ] ) ]} s '
                       f'given by average SPIKE-sync of {np.mean( y_smooth[ ps[ 0 ]:ps[ 1 ] ] )} '
                       f'(std={np.std( y_smooth[ ps[ 0 ]:ps[ 1 ] ] )}, '
                       f'max={np.max( y_smooth[ ps[ 0 ]:ps[ 1 ] ] )}) ' )
        
        f2_ax1.set_xlim( 0, sim_t_array[ -1 ] )
        f2_ax1.set_ylim( 0, 1 )
        f2_ax1.set_xlabel( 'Time (s)', size=axis_label_size, )
        f2_ax1.set_ylabel( 'SPIKE-sync', size=axis_label_size, )
        # f2_ax1.legend( loc='upper right' )
    
    # TODO export PSs to txt file
    # TODO make one subplot of x and u for each attractor
    axes_to_annotate = [ f0_ax1, f1_ax1, f2_ax1 ]
    for ax in axes_to_annotate:
        ax.set_prop_cycle( None )
        # -- add generic stimulus shading
        # TODO multiple generic stimuli
        if generic_stimulus:
            ax.axvspan(
                    generic_stimulus[ 1 ][ 0 ],
                    generic_stimulus[ 1 ][ 1 ],
                    facecolor='grey',
                    alpha=alpha2,
                    )
            ax.annotate( 'GS',
                         xycoords='data',
                         xy=((generic_stimulus[ 1 ][ 0 ] + generic_stimulus[ 1 ][ 1 ]) / 2, 0),
                         xytext=(0, -15), textcoords='offset points',
                         horizontalalignment='right', verticalalignment='bottom',
                         color='grey' )
            if attractors:
                for atr in attractors:
                    x, y, y_smooth, pss = find_ps( path, sim_t_array[ -1 ], atr[ 1 ] )
                    
                    color = next( ax._get_lines.prop_cycler )[ 'color' ]
                    for ps in pss:
                        ax.annotate( 'PS',
                                     xycoords='data',
                                     xy=(x[ ps[ 0 ] + np.argmax( y_smooth[ ps[ 0 ]:ps[ 1 ] ] ) ], ax.get_ylim()[ 1 ]),
                                     horizontalalignment='right', verticalalignment='bottom',
                                     color=color )
    
    # finishing
    
    plt.xlabel( 'time (s)', size=axis_label_size )
    
    fig.suptitle( 'Attractor Neurons\n' + title_addition, fontsize=font_size1 )
    
    if not filename:
        filename = 'x_u_spks_from_basin.png'
    else:
        filename = filename + '.png'
    
    fig.savefig(
            os.path.join( path, filename ),
            bbox_inches='tight' )
    
    if show:
        plt.show()
    
    return fig
