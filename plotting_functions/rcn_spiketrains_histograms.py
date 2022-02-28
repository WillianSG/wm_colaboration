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

import os, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from brian2 import mV, Hz, second, ms, mean, std
from helper_functions.firing_rate_histograms import firing_rate_histograms


def plot_rcn_spiketrains_histograms(
        Einp_spks,
        Einp_ids,
        stim_E_size,
        E_pop_size,
        Iinp_spks,
        Iinp_ids,
        stim_I_size,
        I_pop_size,
        E_spks,
        E_ids,
        I_spks,
        I_ids,
        t_run,
        path_to_plot,
        bin_width_desired=50 * ms,
        show=True):
    # General plotting settings
    lwdth = 3
    s1 = 20
    s2 = 30
    mpl.rcParams['axes.linewidth'] = lwdth

    plt.close('all')

    # Calculating firing rate histograms for plotting

    # 'time_resolved' histograms - [?]
    [input_e_t_hist_count,
     input_e_t_hist_edgs,
     input_e_t_hist_bin_widths,
     input_e_t_hist_fr] = firing_rate_histograms(
        tpoints=Einp_spks,
        inds=Einp_ids,
        bin_width=bin_width_desired,
        N_pop=stim_E_size,
        flag_hist='time_resolved')

    [input_i_t_hist_count,
     input_i_t_hist_edgs,
     input_i_t_hist_bin_widths,
     input_i_t_hist_fr] = firing_rate_histograms(
        tpoints=Iinp_spks,
        inds=Iinp_ids,
        bin_width=bin_width_desired,
        N_pop=stim_I_size,
        flag_hist='time_resolved')

    [e_t_hist_count,
     e_t_hist_edgs,
     e_t_hist_bin_widths,
     e_t_hist_fr] = firing_rate_histograms(
        tpoints=E_spks,
        inds=E_ids,
        bin_width=bin_width_desired,
        N_pop=stim_E_size,
        flag_hist='time_resolved')

    [i_t_hist_count,
     i_t_hist_edgs,
     i_t_hist_bin_widths,
     i_t_hist_fr] = firing_rate_histograms(
        tpoints=I_spks,
        inds=I_ids,
        bin_width=bin_width_desired,
        N_pop=stim_I_size,
        flag_hist='time_resolved')

    # Plotting spiking activity and histograms

    fig = plt.figure(figsize=(20, 15))

    gs = gridspec.GridSpec(9, 1,
                           height_ratios=[1, 0.5, 1, 1, 0.5, 4, 1, 0.5, 1])
    gs.update(wspace=0.07, hspace=0.3, left=None, right=None)

    # Input_I population: histogram
    ax0 = fig.add_subplot(gs[0, 0])

    plt.bar(input_i_t_hist_edgs[:-1], input_i_t_hist_fr, input_i_t_hist_bin_widths, edgecolor='grey', color='white',
            linewidth=lwdth)

    plt.ylabel('$\\nu_{Input I}$\n(Hz)', size=s1, labelpad=35,
               horizontalalignment='center')
    plt.xlim(0, t_run / second)
    plt.ylim(0, 5500)

    ax0.set_yticks(np.arange(0, 5510, 2500))
    ax0.set_xticklabels([])

    plt.tick_params(axis='both', which='major', width=lwdth, length=10, pad=10)
    plt.yticks(size=s1)
    plt.xticks(size=s1)

    # Inhibitory population: spiking
    ax1 = fig.add_subplot(gs[2, 0])

    plt.plot(I_spks, I_ids, '.', color='red')
    plt.ylabel('Source\nneuron $I$', size=s1, labelpad=105, horizontalalignment='center')

    ax1.set_yticks(np.arange(0, I_pop_size + 1, stim_I_size))
    ax1.set_xticklabels([])

    plt.tick_params(axis='both', which='major', width=lwdth, length=10, pad=10)
    plt.ylim(0, I_pop_size)
    plt.yticks(size=s1)
    plt.xticks(size=s1)
    plt.xlim(0, t_run / second)

    # Inhibitory population: histogram
    ax2 = fig.add_subplot(gs[3, 0])

    plt.bar(i_t_hist_edgs[:-1], i_t_hist_fr, i_t_hist_bin_widths, edgecolor='red', color='white', linewidth=lwdth)
    plt.ylabel('$\\nu_{I}$\n(Hz)', size=s1, labelpad=35, horizontalalignment='center')
    plt.xlim(0, t_run / second)
    plt.ylim(0, max(i_t_hist_fr) * 1.1)

    yticks = np.linspace(0, max(i_t_hist_fr) * 1.1, 3)

    ax2.set_yticks(yticks)
    ax2.set_yticklabels(np.around(yticks))
    ax2.set_xticklabels([])

    plt.tick_params(axis='both', which='major', width=lwdth, length=10,
                    pad=10)
    plt.yticks(size=s1)
    plt.xticks(size=s1)

    # Excitatory population: spiking
    ax3 = fig.add_subplot(gs[5, 0])

    plt.plot(E_spks, E_ids, '.', color='mediumblue')
    plt.ylabel('Source neuron $E$', size=s1, labelpad=35, horizontalalignment='center')

    ax3.set_yticks(np.arange(0, E_pop_size + 1, stim_E_size))
    ax3.set_xticklabels([])

    plt.tick_params(axis='both', which='major', width=lwdth, length=10, pad=10)
    plt.ylim(0, E_pop_size)
    plt.yticks(size=s1)
    plt.xticks(size=s1)
    plt.xlim(0, t_run / second)

    # Excitatory population: histogram
    ax4 = fig.add_subplot(gs[6, 0])

    plt.bar(e_t_hist_edgs[:-1], e_t_hist_fr, e_t_hist_bin_widths,
            edgecolor='mediumblue', color='white', linewidth=lwdth)
    plt.ylabel('$\\nu_{E}$\n(Hz)', size=s1, labelpad=35,
               horizontalalignment='center')
    plt.xlim(0, t_run / second)

    yticks = np.linspace(0, max(e_t_hist_fr) * 1.1, 3)

    ax4.set_yticks(yticks)
    ax4.set_yticklabels(np.around(yticks))
    ax4.set_xticklabels([])

    plt.tick_params(axis='both', which='major', width=lwdth, length=10,
                    pad=10)
    plt.yticks(size=s1)
    plt.xticks(size=s1)

    # Input_E population: histogram
    ax5 = fig.add_subplot(gs[8, 0])

    plt.bar(input_e_t_hist_edgs[:-1], input_e_t_hist_fr,
            input_e_t_hist_bin_widths,
            edgecolor='grey',
            color='white',
            linewidth=lwdth)
    plt.ylabel('$\\nu_{Input E}$\n(Hz)', size=s1, labelpad=35, horizontalalignment='center')
    plt.xlim(0, t_run / second)
    plt.ylim(0, 4500)

    ax5.set_yticks(np.arange(0, 4500, 2000))

    plt.yticks(size=s1)
    plt.xticks(size=s1)
    plt.tick_params(axis='both', which='major', width=lwdth, length=10,
                    pad=15)
    plt.xlabel('Time (s)', size=s1)

    if show:
        plt.show()

    fig.savefig(
        path_to_plot + '/rcn_population_spiking.png',
        bbox_inches='tight')
