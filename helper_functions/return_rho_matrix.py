# -*- coding: utf-8 -*-
"""
@author: slehfeldt with adaptations from asonntag

Input:
syn_obj_name: string with name of synapse object
net_obj: network object (example: n.net)
n_src: number of neurons in source population
n_tar: number of neurons in target population

Output:
-  rho_matrix = array with lists of efficacies, each list belongs to one
    source neuron, weights within each list = weights of source neuron onto
    target neurons

Comments:
- returns the synaptic efficacy (rho) matrix of a 
    specified synapse object. Original version of function taken from Steve Nease 
    (get_w_matrix() in rcn project)
"""

def return_rho_matrix(syn_obj_name, net_obj, n_src, n_tar):
	import numpy as np

	rho_matrix = np.zeros(shape=(n_src, n_tar))

	rho_data = net_obj.get_states()[syn_obj_name]

	rho_matrix[rho_data['i'], rho_data['j']] = rho_data['rho']

	return rho_matrix