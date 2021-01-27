# -*- coding: utf-8 -*-
"""
@author: slehfeldt

Inputs:
- t_start: start piont with *second
- t_end: end piont with *second
- inds: array with neuron indices
- tpoints: array with time points of spikes

Outputs:
- inds_pos: neuron indices at the positions belonging to the temporal window
- tpoints_pos: spike times at the positions belonging to the temporal window

Comments:
- Extracts neuron indices and belonging time points of spikes within a temporal window (defined by t_start and t_end).
- Original version of function by Steve Nease (see rcn project).
"""

def extract_trial_data(t_start, t_end, inds, tpoints):
	import numpy as np

	# Find index positions within trial time 
	pos = np.logical_and(tpoints >= t_start,tpoints <=t_end)
	inds_pos = inds[pos]
	tpoints_pos = tpoints[pos]

	return inds_pos, tpoints_pos