# -*- coding: utf-8 -*-
"""
@author: wgirao
@based-on: asonntag
"""

# Removes spike that are far apart by less than a simulation time step (dt = 0.0001 sconds)
"""
input:
- spikes_t(np.array): sorted spike times
- min_dist: the minimum distance between spikes (simulation time step resolution = 0.0001 sconds)
- t_run: simulation time

output:
- spikes_t: original spike times with spikes separated by less than 'min_dist' (spikes in the same timestep) removed.

Comments:
- Annkatherin originally used 7 for the precision variable (why?).
- if abs(spikes_t[index - shift] - prev_timestamp) < min_dist it means that the interspike interval is smaller than a simulation time step.
"""
def remove_close_spikes(spikes_t, min_dist, t_run):
	shift = 0 # ?
	prev_timestamp = 0 # loop variable

	# ALTERED (E) - precision defined by amount of decimal places available for the simulation time step (time resolution)
	precision = len(str(min_dist).split('.',1)[1]) # Times must differ at least on the final decimal place of 'min_dist'

	for index in range(0, len(spikes_t)):
		spikes_t[index - shift] = round(spikes_t[index - shift], precision)

		if round((spikes_t[index - shift] - prev_timestamp), precision) < min_dist or spikes_t[index - shift] > t_run:
			spikes_t[index - shift] = 0
		else:
			prev_timestamp = spikes_t[index - shift]

	return spikes_t