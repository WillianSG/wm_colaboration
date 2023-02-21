# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System
"""
import numpy as np

"""
Counts the number of population spikes (PS) for each attractor during the whole simulation course.

Return: dictionary with attractor ID as key an PS counts as value (e.g. {'A1': {'triggered': 4, 'spontaneous': 3}}).

triggered: count of PSs happening inside a time window starting after the cuing of the attractor and ending at the start of a second attractor cuing (or end of simulation).

spontaneous: count of PSs happening inside a time window where a different cued attractor should be active. Obs.: The count assumes that a currently active attractor gets deactivated once a different attractor is cued.
"""


def count_ps(rcn, attractor_cueing_order):
    from helper_functions.other import find_ps

    # needed to output the spikes to file, or they won't be found by find_ps()
    rcn.get_spikes_pyspike()

    attractor_ps_counts = {}

    us_neurs_with_input, sim_t_array, U, tau_f = rcn.get_u_traces_from_pattern_neurons()

    attractor_times = {}
    for a in attractor_cueing_order:
        if not a[0] in attractor_times:
            attractor_times[a[0]] = [(a[3][2][0], a[3][2][1], a[3][2][1] + a[2])]
        else:
            attractor_times[a[0]].append((a[3][2][0], a[3][2][1], a[3][2][1] + a[2]))

    for a in attractor_cueing_order:
        if not a[0] in attractor_ps_counts:
            attractor_ps_counts[a[0]] = {'triggered': list(), 'spontaneous': list()}

        x, y, y_smooth, pss = find_ps(rcn.net_sim_data_path, sim_t_array[-1], a)

        # check each PS if it is triggered or spontaneous (4 cases)
        for ps in pss:
            max_sync_t = x[ps[0] + np.argmax(y_smooth[ps[0]:ps[1]])]
            # if PS in triggered window of this cue
            if a[3][2][1] <= max_sync_t <= a[3][2][1] + a[2]:
                attractor_ps_counts[a[0]]['triggered'].append(max_sync_t)
            # if PS during this cue
            elif a[3][2][0] <= max_sync_t <= a[3][2][1]:
                continue
            # if PS outside activity window of this cue
            elif (max_sync_t < a[3][2][0] or max_sync_t > a[3][2][1] + a[2]):
                # if PS during another cue's activity window
                for at in attractor_times[a[0]]:
                    if at[0] <= max_sync_t <= at[2]:
                        break
                # finally, if PS is not during another cue's activity window
                else:
                    # the first pass of this loop will count the spontaneous PSs while the others should avoid doing so or they'll be counted twice
                    if max_sync_t not in attractor_ps_counts[a[0]]['spontaneous']:
                        attractor_ps_counts[a[0]]['spontaneous'].append(max_sync_t)

    return attractor_ps_counts


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
