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

	from other import find_ps

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

					for a in range(j+1, len(x)):

						if y[a] < 0.3:

							_last_trg_ps = x[a]

							break

			else:

				if (x[j] < time_window[i][0] or x[j] > time_window[i][1]) and (x[j] < time_window[i][2][0] or x[j] > time_window[i][2][1]):

					if y[j] >= spk_sync_thr and x[j] > _last_spt_ps:

						attractors_ps_counts[attractors[i][0]]['spontaneous'] += 1

						for a in range(j+1, len(x)):

							if y[a] < 0.3:

								_last_spt_ps = x[a]

								break

	return attractors_ps_counts

def count_ps_v2(rcn, attractors: list, time_window: list, spk_sync_thr: float):

	from other import find_ps

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

					_last_trg_ps = x[j] + 0.15

			else:

				if (x[j] < time_window[i][0] or x[j] > time_window[i][1]) and (x[j] < time_window[i][2][0] or x[j] > time_window[i][2][1]):

					if y[j] >= spk_sync_thr and x[j] > _last_spt_ps:

						attractors_ps_counts[attractors[i][0]]['spontaneous'] += 1

						_last_spt_ps = x[j] + 0.15

	return attractors_ps_counts

	