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
from helper_functions.other import make_folders


def plot_x_u_spks_from_basin( path, generic_stimulus=None, filename=None, title_addition='', show=True ):
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
    heights = [ 2, 2 ]
    
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
    
    plt.ylim( 0.0, 1.0 )
    plt.xlim( 0.0, sim_t_array[ -1 ] )
    
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
    
    plt.ylim( 0.0, 1.0 )
    plt.xlim( 0.0, sim_t_array[ -1 ] )
    
    # plt.axvspan(
    #         0.0,
    #         (stim_pulse_duration / ms) / 1000,
    #         facecolor='grey',
    #         alpha=alpha2,
    #         label='PS (' + str( stim_pulse_duration ) + ')' )
    if generic_stimulus:
        plt.axvspan(
                generic_stimulus[ 1 ][ 0 ],
                generic_stimulus[ 1 ][ 1 ],
                facecolor='grey',
                alpha=alpha2,
                )
    
    # plt.legend( loc='upper right', fontsize=font_size2 )
    
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels( [ ] )
    
    # plt.xticks( np.arange(
    #         0.0,
    #         sim_t_array[ -1 ] + 0.5,
    #         step=0.5 ) )
    
    # spks
    f1_ax1 = fig.add_subplot( spec2[ 1, 0 ] )
    
    f1_ax1.plot( spk_mon_ts, spk_mon_ids, '|', color='k', zorder=0 )
    plt.xlim( 0.0, sim_t_array[ -1 ] )
    
    if generic_stimulus:
        f1_ax1.axvspan(
                generic_stimulus[ 1 ][ 0 ],
                generic_stimulus[ 1 ][ 1 ],
                facecolor='grey',
                alpha=alpha2,
                )
    
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
