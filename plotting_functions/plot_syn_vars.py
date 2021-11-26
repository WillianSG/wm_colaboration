# -*- coding: utf-8 -*-


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from brian2 import second

"""
@author: t.f.tiotto@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System

Function:
-

Script arguments:
-

Script output:
-
"""


def plot_syn_vars(path, spiked_neurons, synapse, synapse_monitor, window, show=False):
    # get indices of neurons which have spiked
    spiked_idx = np.where(spiked_neurons)[0]
    # get indices of timepoints within the window
    timepoints = np.where((synapse_monitor.t > window[0]) & (synapse_monitor.t < window[1]))[0]
    # extract synaptic variables and filter them
    us = synapse_monitor[synapse[spiked_idx, :]].u[:, timepoints]
    xs = synapse_monitor[synapse[spiked_idx, :]].x_[:, timepoints]

    t = synapse_monitor.t[timepoints]

    # General plotting settings
    lwdth = 3
    s1 = 20
    s2 = 30
    mpl.rcParams['axes.linewidth'] = lwdth
    plt.close('all')
    fig = plt.figure(figsize=(32, 22), constrained_layout=True)
    gs = mpl.gridspec.GridSpec(nrows=4, ncols=5,
                               hspace=0.1,
                               # wspace=0.2,
                               height_ratios=[1, 1, 1.5, 1.5],
                               figure=fig)
    gs.update(top=0.965, bottom=0.03)
    fig.suptitle(
        f'Mean calcium and resources for synapses of active neurons in t=[{window[0] * second}-{window[1] * second}] s',
        size=s2)

    neuron_sample = np.random.choice(spiked_idx, size=5, replace=False)
    for i, s in enumerate(neuron_sample):
        ui = np.mean(synapse_monitor[synapse[s, :]].u[:, timepoints], axis=0)
        ax0 = fig.add_subplot(gs[0, i])
        ax0.plot(t / second, ui, color='blue')
        ax0.set_ylabel('Mean calcium ($\\bar{u}$)', size=s1, labelpad=5, horizontalalignment='center')
        ax0.set_title(f'Neuron {s}', fontsize=s1)
        xi = np.mean(synapse_monitor[synapse[s, :]].x_[:, timepoints], axis=0)
        ax1 = fig.add_subplot(gs[1, i])
        ax1.plot(t / second, xi, color='orange')
        ax1.set_ylabel('Mean resources ($\\bar{x}$)', size=s1, labelpad=5, horizontalalignment='center')

    def confidence_interval(data, cl=0.95):
        degrees_freedom = len(data) - 1
        sampleMean = np.mean(data, axis=0)
        sampleStandardError = st.sem(data, axis=0)
        confidenceInterval = st.t.interval(alpha=cl, df=degrees_freedom, loc=sampleMean,
                                           scale=sampleStandardError)
        return confidenceInterval

    ci_us = confidence_interval(us)
    ax2 = fig.add_subplot(gs[2, :])
    ax2.plot(t / second, np.mean(us, axis=0), linewidth=1, color='blue', label='Mean')
    ax2.fill_between(t / second, ci_us[0], ci_us[1], facecolor='blue', alpha=0.5, label='95% CI')
    ax2.set_title('All neurons', fontsize=s1)
    ax2.set_ylabel('Mean calcium ($\\bar{u}$)', size=s1, labelpad=5, horizontalalignment='center')
    ax2.legend()

    ci_xs = confidence_interval(xs)
    ax3 = fig.add_subplot(gs[3, :])
    ax3.plot(t / second, np.mean(xs, axis=0), linewidth=1, color='orange', label='Mean')
    ax3.fill_between(t / second, ci_xs[0], ci_xs[1], facecolor='orange', alpha=0.5, label='95% CI')
    ax3.set_ylabel('Mean resources ($\\bar{x}$)', size=s1, labelpad=5, horizontalalignment='center')
    ax3.legend()

    for ax in fig.get_axes():
        ax.tick_params(axis='both', which='major', labelsize=s1, width=lwdth, length=10, pad=15)
    plt.xlabel('Time (s)', size=s1)

    if show:
        plt.show()

    fig.savefig(
        path + f'/rcn_syn_variables_{window[0] * second}-{window[1] * second}-s.png',
        bbox_inches='tight')


"""
@author: t.f.tiotto@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System

Function:
-

Script arguments:
-

Script output:
-
"""


def plot_membrane_potentials(path, spiked_neurons, neuron_monitor, window, show=False):
    # get indices of neurons which have spiked
    spiked_idx = np.where(spiked_neurons)[0]
    # randomly sample 10 spiked neurons
    neuron_sample = np.random.choice(spiked_idx, size=10, replace=False)
    # get indices of timepoints within the window
    timepoints = np.where((neuron_monitor.t > window[0]) & (neuron_monitor.t < window[1]))[0]
    # extract synaptic variables and filter them
    Vepsps = neuron_monitor[neuron_sample].Vepsp[:, timepoints]
    t = neuron_monitor.t[timepoints]

    # General plotting settings
    lwdth = 3
    s1 = 20
    s2 = 30
    mpl.rcParams['axes.linewidth'] = lwdth

    plt.close('all')

    fig = plt.figure(figsize=(32, 22), constrained_layout=True)
    gs = mpl.gridspec.GridSpec(nrows=3, ncols=5,
                               hspace=0.1,
                               # wspace=0.2,
                               height_ratios=[1, 1, 1.5],
                               figure=fig)
    gs.update(top=0.965, bottom=0.03)
    fig.suptitle(f'Calcium and resources for active neurons in t=[{window[0] * second}-{window[1] * second}] s',
                 size=s2)

    idx = np.unravel_index(range(len(neuron_sample)), (2, 5))
    for k, (i, j) in enumerate(zip(idx[0], idx[1])):
        ax0 = fig.add_subplot(gs[i, j])
        ax0.plot(t / second, Vepsps[k], color='green')
        ax0.set_ylabel(f'Vepsp', size=s1, labelpad=5, horizontalalignment='center')
        ax0.set_title(f'Neuron {neuron_sample[k]}', fontsize=s1)

    def confidence_interval(data, cl=0.95):
        degrees_freedom = len(data) - 1
        sampleMean = np.mean(data, axis=0)
        sampleStandardError = st.sem(data, axis=0)
        confidenceInterval = st.t.interval(alpha=cl, df=degrees_freedom, loc=sampleMean,
                                           scale=sampleStandardError)
        return confidenceInterval

    ci_us = confidence_interval(Vepsps)
    ax2 = fig.add_subplot(gs[2, :])
    ax2.plot(t / second, np.mean(Vepsps, axis=0), linewidth=1, color='green', label='Mean')
    ax2.fill_between(t / second, ci_us[0], ci_us[1], facecolor='green', alpha=0.5, label='95% CI')
    ax2.set_title('All neurons', fontsize=s1)
    ax2.set_ylabel('Mean calcium ($\\bar{u}$)', size=s1, labelpad=5, horizontalalignment='center')
    ax2.legend()

    for ax in fig.get_axes():
        ax.tick_params(axis='both', which='major', labelsize=s1, width=lwdth, length=10, pad=15)
    plt.xlabel('Time (s)', size=s1)

    if show:
        plt.show()

    fig.savefig(
        path + f'/rcn_Vepsp_{window[0] * second}-{window[1] * second}-s.png',
        bbox_inches='tight')
