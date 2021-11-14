# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System
@based-on: "Synaptic Plasticity in Spiking Neural Networks - Annkathrin SONNTAG (Master Thesis, 2020)"

Function:
- Removes spike that are far apart by less than a simulation time step (dt = 0.0001 sconds).

Script arguments:
-

Script output:
- spikes_t: original spike times with spikes separated by less than 'min_dist' (spikes in the same timestep) removed.

Comments:
- Annkatherin originally used 7 for the precision variable (why?).
- if abs(spikes_t[index - shift] - prev_timestamp) < min_dist it means that the interspike interval is smaller than a simulation time step.

Inputs:
- spikes_t(np.array): sorted spike times
- min_dist: the minimum distance between spikes (simulation time step resolution = 0.0001 sconds)
- t_run: simulation time
"""
import numpy as np 

def check_spikes_isi(spikes_t, min_dist):
	prev_timestamp = 0 # loop variable

	# Maybe shoudl stop rounding
	for index in range(0, len(spikes_t)):
		if spikes_t[index] < 0:
			raise ValueError("negaive spike detected")
		elif (spikes_t[index] - prev_timestamp) <= min_dist:
			spikes_t[index] = spikes_t[index] + min_dist

		prev_timestamp = spikes_t[index]

	return spikes_t