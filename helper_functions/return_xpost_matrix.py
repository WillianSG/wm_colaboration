# -*- coding: utf-8 -*-
"""
@author: slehfeldt with adaptations from asonntag

Input:
- syn_obj_name: string with name of synapse object.
- net_obj: network object (example: n.net).
- n_src: number of neurons in source population.
- n_tar: number of neurons in target population.

Output:
- xpost_matrix = array with lists of weights, each list belongs to one source neuron, weights within each list = weights of source neuron onto target neurons.

Comments:
- returns the weight matrix of a specified synapse object. Original version of function taken from Steve Nease (get_xpost_matrix() in rcn project).
"""

def return_xpost_matrix(syn_obj_name, net_obj, n_src, n_tar):
	from brian2 import mV
	import numpy as np


	xpost_matrix = np.zeros(shape=(n_src, n_tar))

	xpost_data = net_obj.get_states()[syn_obj_name]

	xpost_matrix[xpost_data['i'], xpost_data['j']] = xpost_data['xpost']

	return xpost_matrix