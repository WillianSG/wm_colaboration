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

Comments:
"""

'''
This function takes the spikes vector and it checks for every spike pair the distance between them,
If the distance is too close it set the second one to zero
:return: spikes
'''

def remove_close_spikes(spikes_t, min_dist, t_run):
	shift = 0 # ?
	precision = 7 # decimal value of the time

	prev_timestamps = 0 # loop variable		
	for index in range(0, len(spikes_t)):
		print(spikes_t[index - shift])
		spikes_t[index - shift] = round(spikes_t[index - shift], precision)
		print(spikes_t[index - shift])
		print('\n')

		if abs(spikes_t[index - shift] - prev_timestamps) < min_dist or spikes_t[index - shift] > t_run:
			spikes_t[index - shift] = 0
		else:
			prev_timestamps = spikes_t[index - shift]

	return spikes_t