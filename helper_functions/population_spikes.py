# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System
"""

"""
Counts the number of population spikes (PS) for each attractor during the whole simulation course.

Return: dictionary with attractor ID as key an PS counts as value (e.g. {'A1': {'triggered': 4, 'spontaneous': 3}}).

triggered: count of PSs happening inside a time window starting after the cuing of the attractor and ending at the start of a second attractor cuing (or end of simulation).

spontaneous: count of PSs happening inside a time window where a different cued attractor should be active. Obs.: The count assumes that a currently active attractor gets deactivated once a different attractor is cued.
"""


def count_ps(rcn, attractors: list, time_window: list, spk_sync_thr: float):
    from helper_functions.other import find_ps

    # needed to output the spikes to file or they won't be found bt find_ps()
    rcn.get_spikes_pyspike()

    attractors_ps_counts = {}

    us_neurs_with_input, sim_t_array, U, tau_f = rcn.get_u_traces_from_pattern_neurons()

    for i in range(0, len(attractors)):

        attractors_ps_counts[attractors[i][0]] = {'triggered': 0, 'spontaneous': 0}

        x, y, y_smooth, pss = find_ps(rcn.net_sim_data_path, sim_t_array[-1], attractors[i])

        _last_trg_ps = 0
        _last_spt_ps = 0

        for j in range(0, len(x)):

            if x[j] >= time_window[i][0] and x[j] <= time_window[i][1]:

                if y[j] >= spk_sync_thr and x[j] > _last_trg_ps:
                    attractors_ps_counts[attractors[i][0]]['triggered'] += 1
                    # TODO check if this is correct.  I think always adding 0.15s could lead to issues.  Try my alternative commented out line below
                    _last_trg_ps = x[j] + 0.15
                    # _last_trg_ps = x[j] + (x[pss[0][1]] - x[pss[0][0]])

            else:

                if (x[j] < time_window[i][0] or x[j] > time_window[i][1]) and (
                        x[j] < time_window[i][2][0] or x[j] > time_window[i][2][1]):

                    if y[j] >= spk_sync_thr and x[j] > _last_spt_ps:
                        attractors_ps_counts[attractors[i][0]]['spontaneous'] += 1

                        _last_spt_ps = x[j] + 0.15
                        # _last_spt_ps = x[j] + (x[pss[0][1]] - x[pss[0][0]])

    return attractors_ps_counts


"""
"""


def find_gap_between_PS(rcn, attractor: list, spk_sync_thr: float):
    from other import find_ps
    import numpy as np

    us_neurs_with_input, sim_t_array, U, tau_f = rcn.get_u_traces_from_pattern_neurons()

    x, y, y_smooth, pss = find_ps(rcn.net_sim_data_path, sim_t_array[-1], attractor[0])

    psStart = -1
    psEnd = -1
    ps1st = True

    psStart2nd = -1
    psEnd2nd = -1

    for i in range(len(y)):

        if y[i] >= spk_sync_thr and ps1st:

            psStart = x[i]

        elif (psStart != -1) and (y[i] < spk_sync_thr - 0.2) and (x[i] > psStart) and ps1st:

            psEnd = x[i]

            ps1st = False

        else:

            pass

        if psEnd != -1:

            if y[i] >= spk_sync_thr and x[i] > psEnd:

                psStart2nd = x[i]

            elif (psStart2nd != -1) and (y[i] < spk_sync_thr - 0.2) and (x[i] > psStart2nd):

                psEnd2nd = x[i]

            else:

                pass

        if psEnd != -1 and psEnd2nd != -1:
            return np.round(psEnd, 3), np.round(psStart2nd, 3)
