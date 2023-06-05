# -*- coding: utf-8 -*-
"""
@author: t.f.tiotto@rug.nl / w.soares.girao@rug.nl
@university: University of Groningen
@group: Bernoulli Institute / Bio-Inspired Circuits and System
"""
import collections
import os, sys, pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from brian2 import second, ms, units
import numpy as np

if sys.platform == 'linux':

    root = os.path.dirname(os.path.abspath(os.path.join(__file__, '../')))

    sys.path.append(os.path.join(root, 'helper_functions'))

    from other import *

else:

    from helper_functions.other import *


def plot_x_u_spks_from_basin(path, attractor_cues=None, pss_categorised=None, rcn=None,
                             filename=None, num_neurons=None, title_addition='', show=True):
    # Plot settings
    plt.close('all')

    axis_label_size = 10
    suptitle_fontsize = 14
    title_fontsize = 12

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
    colour_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.rcParams['axes.titlepad'] = 20

    # Extract name and neuron IDs of attractors
    attractors_unique = {}
    for a in attractor_cues:
        if a[0] not in attractors_unique:
            attractors_unique[a[0]] = a[1]
    attractors_unique = collections.OrderedDict(sorted(attractors_unique.items()))

    widths = [10]
    heights = [2] * len(attractors_unique) + [2, 2, 2]

    fig = plt.figure(figsize=(15, 6 + 1.5 * len(attractors_unique)))

    spec2 = gridspec.GridSpec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        figure=fig)

    # -- Read x and u from synapses and neurons
    if rcn.dumped_mons_dict:
        x_traces, sim_t_array, tau_d = rcn.dumped_mons_dict['x_traces']
        u_traces, sim_t_array, U, tau_f = rcn.dumped_mons_dict['u_traces']
    else:
        x_traces, sim_t_array, tau_d = rcn.get_x_traces_from_pattern_neurons()
        u_traces, sim_t_array, U, tau_f = rcn.get_u_traces_from_pattern_neurons()

    for i, (atr, ids) in enumerate(attractors_unique.items()):
        f_ax1 = fig.add_subplot(spec2[i, 0])

        color = colour_cycle[i]

        xs_atr = []
        for k in x_traces:
            if k[0] in ids:
                xs_atr.append(x_traces[k])
        us_atr = []
        for k in u_traces:
            if k in ids:
                us_atr.append(u_traces[k])

        x_mean = np.mean(xs_atr, axis=0)
        x_std = np.std(xs_atr, axis=0)

        u_mean = np.mean(us_atr, axis=0)
        u_std = np.std(us_atr, axis=0)

        f_ax1.plot(
            sim_t_array,
            x_mean,
            color=x_color,
            zorder=0,
            linewidth=linewidth2)

        f_ax1.fill_between(
            sim_t_array,
            x_mean + x_std,
            x_mean - x_std,
            color=x_color,
            alpha=alpha2)

        f_ax1.set_ylabel(
            'x, u \n(a.u.)',
            size=axis_label_size,
            color='black')

        f_ax1.tick_params(axis='y', labelcolor=x_color)

        f_ax1.set_ylim(0, 1)
        f_ax1.set_xlim(0, sim_t_array[-1])

        f_ax1.tick_params(axis='y', labelcolor=u_color, left=False)

        # 2nd y axis: us
        f_ax2 = f_ax1.twinx()

        f_ax2.plot(
            sim_t_array,
            u_mean,
            zorder=0,
            linewidth=linewidth2,
            color=u_color)

        f_ax2.fill_between(
            sim_t_array,
            u_mean + u_std,
            u_mean - u_std,
            color=u_color,
            alpha=alpha2)

        # f_ax2.set_ylabel(
        #         'u (a.u.)',
        #         size=axis_label_size,
        #         color=u_color )

        f_ax2.tick_params(axis='y', labelcolor=u_color, left=False)

        f_ax2.set_ylim(0, 1)
        f_ax2.set_xlim(0, sim_t_array[-1])

        # 3rd y axis: weights
        def make_patch_spines_invisible(ax):
            ax.set_frame_on(True)
            ax.patch.set_visible(False)
            for sp in ax.spines.values():
                sp.set_visible(False)

        f_ax3 = f_ax2.twinx()

        f_ax3.spines["right"].set_position(("axes", 1))
        make_patch_spines_invisible(f_ax3)
        f_ax3.spines["right"].set_visible(True)

        x_times_u = (x_mean * u_mean) / U
        f_ax3.plot(
            sim_t_array,
            x_times_u,
            zorder=0,
            linewidth=linewidth2,
            color=ux_color)
        f_ax3.set_ylabel(
            'x*u*1/U \n (a.u.)',
            size=axis_label_size,
            color=ux_color)
        f_ax3.tick_params(axis='y', labelcolor=ux_color)
        f_ax3.set_ylim(0, np.max(x_times_u))

        f_ax1.set_title(f'Attractor {atr}', size=title_fontsize, color=color)

    # -------------------------------------------------------------------------
    # ------- Plot thresholds
    # -------------------------------------------------------------------------
    f_thresh = fig.add_subplot(spec2[len(attractors_unique), 0])

    # load dumped data if it exists
    if rcn.dumped_mons_dict:
        x = rcn.dumped_mons_dict['E_rec']['t']
        data = rcn.dumped_mons_dict['E_rec']['Vth_e']
    else:
        E_rec = rcn.E_rec
        x = E_rec.t
        data = E_rec.Vth_e

    # -- plot voltage thresholds --
    for i, (atr, ids) in enumerate(attractors_unique.items()):
        if rcn.dumped_mons_dict:
            y = np.mean(data[:, ids], axis=1)
        else:
            y = np.mean(data[ids, :], axis=0)

        color = colour_cycle[i]

        f_thresh.plot(x, y, label=atr, color=color)
        # f_thresh.set_ylim(np.min(rcn.E_rec.Vth_e), np.max(rcn.E_rec.Vth_e))
        f_thresh.set_xlim(0, sim_t_array[-1])
    f_thresh.set_title('Voltage thresholds', size=title_fontsize)

    # -------------------------------------------------------------------------
    # ------- Plot spikes
    # -------------------------------------------------------------------------
    spk_mon_ids, spk_mon_ts = rcn.get_spks_from_pattern_neurons()

    ax_spikes = fig.add_subplot(spec2[len(attractors_unique) + 1, 0])
    # -- plot neuronal spikes with attractors in different colours
    for i, (atr, ids) in enumerate(attractors_unique.items()):
        spk_mon_ts = np.array(spk_mon_ts)
        spk_mon_ids = np.array(spk_mon_ids)

        # color = next( f2_ax1._get_lines.prop_cycler )[ 'color' ]
        color = colour_cycle[i]

        atr_indexes = np.argwhere(
            np.logical_and(np.array(spk_mon_ids) >= ids[0],
                           np.array(spk_mon_ids) <= ids[-1]))
        atr_spks = spk_mon_ts[atr_indexes]
        atr_ids = spk_mon_ids[atr_indexes]

        ax_spikes.plot(atr_spks, atr_ids, '|', color=color, zorder=0)

    # f0_ax1.set_ylim( 0, n_neurons )
    if num_neurons:
        ax_spikes.set_ylim(0, num_neurons)
    ax_spikes.set_xlim(0, sim_t_array[-1])

    ax_spikes.set_ylabel(
        'Neuron ID',
        size=axis_label_size,
        color='k')

    ax_spikes.set_title(f'Neural spikes', size=title_fontsize)

    # -------------------------------------------------------------------------
    # ------- Plot SPIKE-synchronisation profile
    # -------------------------------------------------------------------------
    f3_ax1 = fig.add_subplot(spec2[len(attractors_unique) + 2, 0])

    for i, (atr, ids) in enumerate(attractors_unique.items()):
        x, y, y_smooth, pss = rcn.find_ps((atr, ids))

        color = colour_cycle[i]

        f3_ax1.plot(x, y, color=color, alpha=0.5, label=atr)
        f3_ax1.plot(x, y_smooth, '.', markersize=0.5, color=color)

    f3_ax1.set_xlim(0, sim_t_array[-1])
    f3_ax1.set_ylim(0, 1)
    f3_ax1.set_xlabel('Time (s)', size=axis_label_size)
    f3_ax1.set_ylabel('SPIKE-sync', size=axis_label_size, )

    f3_ax1.set_title(f'SPIKE-sync profile', size=title_fontsize)

    # f3_ax1.legend( loc='upper right' )

    for ax in fig.get_axes():
        ax.set_prop_cycle(None)

        for i, (atr, ids) in enumerate(attractors_unique.items()):
            if not pss_categorised:
                x, y, y_smooth, pss = rcn.find_ps((atr, ids))

                # color = next( ax._get_lines.prop_cycler )[ 'color' ]
                color = colour_cycle[i]

                if pss.size:
                    for ps in pss:
                        ax.annotate('PS',
                                    xycoords='data',
                                    xy=(x[ps[0] + np.argmax(y_smooth[ps[0]:ps[1]])], ax.get_ylim()[1]),
                                    horizontalalignment='right', verticalalignment='bottom',
                                    color=color)
            else:
                color = colour_cycle[i]

                for ps_trig in pss_categorised[atr]['triggered']:
                    ax.annotate('T',
                                xycoords='data',
                                xy=(ps_trig, ax.get_ylim()[1]),
                                horizontalalignment='right', verticalalignment='bottom',
                                color=color)
                for ps_spont in pss_categorised[atr]['spontaneous']:
                    ax.annotate('S',
                                xycoords='data',
                                xy=(ps_spont, ax.get_ylim()[1]),
                                horizontalalignment='right', verticalalignment='bottom',
                                color=color)

    # -- add generic stimuli shading
    gss = [a[3] for a in attractor_cues if a[3]]
    if gss:
        for gs in gss:
            if gs[0] > 0:
                for g in gs[1]:
                    ax_spikes.fill_between(
                        [gs[2][0], gs[2][1]],
                        g, g,
                        alpha=alpha2,
                        color='grey'
                    )
            ax_spikes.annotate('GS',
                               xycoords='data',
                               xy=((gs[2][0] + gs[2][1]) / 2, 0),
                               xytext=(0, -15), textcoords='offset points',
                               horizontalalignment='right', verticalalignment='bottom',
                               color='grey')

    # finishing

    plt.xlabel('time (s)', size=axis_label_size)

    trig, spont, accuracy, stability, f1_score = compute_ps_score(pss_categorised, attractor_cues)

    fig.suptitle('Attractor Neurons\n' + title_addition, fontsize=suptitle_fontsize)

    fig.subplots_adjust(bottom=0.2)
    plt.figtext(0.1, 0.01,
                f'F1-Score: {f1_score:.2f}',
                ha='center', fontsize=title_fontsize,
                bbox={'facecolor': 'white', 'pad': 5})

    fig.tight_layout()

    if not filename:
        filename = 'x_u_spks_from_basin.png'
    else:
        filename = filename + '.png'

    fig.savefig(
        os.path.join(path, filename),
        bbox_inches='tight')

    if show:
        plt.show()

    return fig
