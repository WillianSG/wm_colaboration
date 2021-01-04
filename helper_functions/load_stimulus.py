# -*- coding: utf-8 -*-
"""
@author: slehfeldt

Input:
- stimulus_type: string describing the stimulus type ['square', 'circle', 'triangle', 'cross', 'random, 'flat_to_I', 'flat_to_E' and 'flat_to_E_fixed_size']
- stimulus_size: (?)
- offset: (?)

Output:
- Returns a list of input neuron indices that are active to form a stimulus pattern.

Comments:
"""

import random
import numpy as np

def load_stimulus(stimulus_type, stimulus_size, offset):
	stimulus_inds = []

	if stimulus_type == 'square':
		# Indices of input neurons active to form a square
		stimulus_inds = [51,52,53,54,55,56,57,58,59,60,67,68,69,70,71,72,73,74,
			75,76,83,84,85,90,91,92,99,100,107,108,115,116,123,124,131,132,139,
			140,147,148,155,156,163,164,165,170,171,172,179,180,181,182,183,
			184,185,186,187,188,195,196,197,198,199,200,201,202,203,204]

	if stimulus_type == 'circle':
		# Indices of input neurons active to form a circle
		stimulus_inds = [37,38,39,40,41,42,52,53,54,55,56,57,58,59,67,68,69,74,
			75,76,82,83,84,91,92,93,98,99,108,109,114,115,124,125,130,131,140,
			141,146,147,156,157,162,163,164,171,172,173,179,180,181,186,187,
			188,196,197,198,199,200,201,202,203,213,214,215,216,217,218]

	if stimulus_type == 'triangle':
		# Indices of input neurons active to form a triangle
		stimulus_inds = [33,34,35,36,37,38,39,40,41,42,43,44,45,46,50,51,52,53,
			54,55,56,57,58,59,60,61,67,68,75,76,83,84,85,90,91,92,100,101,106,
			107,116,117,118,121,122,123,133,134,137,138,149,150,151,152,153,
			154,166,167,168,169,182,183,184,185,199,200,215,216]

	if stimulus_type == 'cross':
		# Indices of input neurons active to form a cross
		stimulus_inds = [34,35,36,43,44,45,50,51,52,59,60,61,67,68,69,74,75,76,
			84,85,86,89,90,91,101,102,103,104,105,106,118,119,120,121,134,135,
			136,137,149,150,151,152,153,154,164,165,166,169,170,171,179,180,
			181,186,187,188,194,195,196,203,204,205,210,211,212,219,220,221]

	if stimulus_type == 'random':
		stimulus_size = 255 # Size of stimulus = input cell indices
		num_active = 110 # Number of active input neurons = length of stimulus_inds
		stimulus_inds=random.sample(range(0, stimulus_size), num_active)               

	if stimulus_type == 'flat_to_I':
		stimulus_inds = np.arange(0, stimulus_size, 1)

	if stimulus_type == 'flat_to_E':
		stimulus_inds = np.arange(0, stimulus_size, 1)

	if stimulus_type == 'flat_to_E_fixed_size':
		stimulus_inds=np.arange(offset, offset + stimulus_size, 1)

	return stimulus_inds


