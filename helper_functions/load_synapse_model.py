# -*- coding: utf-8 -*-
"""
@author: wgirao
@based-on: asonntag
"""

# Creates the differential equations used to describe the synaptic model of weight (w) update as argument for Brian2's Synapses() method.
"""
input:
- plasticity_rule (str): rule's name
- neuron_type (str): spiking activity
- bistability (bool): with or without it

output:
- model_E_E: synaptic model equation
- pre_E_E: on presy- spike
- post_E_E: on postsy- spike

Comments:
- As the weight (in units of volts) updated depends on other admentional variables, 'rho' is used as a dimensionless variable that helps to calculate a weight update in volts.
"""
from brian2 import *

def load_synapse_model(plasticity_rule, neuron_type, bistability):
	# Non-plastic synapse
	model_E_E_non_plastic = '''w : volt'''

	# Plastic synapse (STDP)
	if bistability == True:
		model_E_E_plastic = ''' 
		w : volt
		dxpre/dt = -xpre / tau_xpre : 1 (clock-driven)
		dxpost/dt = -xpost / tau_xpost : 1 (clock-driven) 
		drho/dt = (int(rho > thr_b_rho)*int(rho < rho_max)  * alpha -
		int(rho <= thr_b_rho) * beta * int(rho > rho_min)) / tau_rho 
			: 1  (clock-driven)'''
	elif bistability == False:
		model_E_E_plastic = ''' 
		w : volt
		rho : 1
		dxpre/dt = -xpre / tau_xpre : 1 (clock-driven)
		dxpost/dt = -xpost / tau_xpost : 1 (clock-driven) '''
	else:
		print ('Bistabilty setting unclear')

	# Learning Rule (weight update)

	# - On post spike (LR1) 
	"""
	xpost_jump: A_post
	xpre_factor: c
	"""
	post_E_E_LR1 = '''xpost = xpost + xpost_jump
		rho = clip(rho + xpre * xpre_factor, rho_min, rho_max)
		w = rho*w_max''' 

	# - On post spike (LR2) 
	"""
	xpost_jump: A_post
	xpre_factor: c
	rho_dep2: rho_neg2
	"""
	post_E_E_LR2 = '''xpost = xpost + xpost_jump
		rho = clip(rho + xpre * xpre_factor+ rho_neg2 *int(xpre < thr_pre) * int(xpre > 0), rho_min, rho_max)
		w = rho*w_max'''

	# - On pre spike (both LR1/LR2) 
	"""
	xpre_jump: A_pre
	rho_dep: rho_neg
	"""
	xpre_update = {'xpre_update': '''xpre = xpre + xpre_jump'''}
	w_update = {'w_update' : ''' w = rho * w_max'''}
	rho_update_pre = {'rho_update_pre':'''rho = clip(rho + rho_neg *int(xpost > thr_post), rho_min, rho_max)'''}

	# Defines the argument 'on_pre' for Brian2 - same as on_pre='v_post += w'
	"""
	Vepsp: effect of presynaptic activity on postsynaptic mem. potential.
	"""
	Vepsp_transmission = {'Vepsp_transmission' : '''Vepsp += w'''}

	# Creaing the equation structure (eqs) needed for Brian2

	# - Neurons without synapses
	if plasticity_rule == 'none' and (neuron_type == 'spikegenerators' or neuron_type == 'poisson'):
		model_E_E = model_E_E_non_plastic
		pre_E_E = ''
		post_E_E = ''

	# - LIF neuron responding to pre- activity with non-plastic synapse
	elif plasticity_rule == 'none' and neuron_type == 'LIF':
		model_E_E = model_E_E_non_plastic
		pre_E_E = dict(Vepsp_transmission)
		post_E_E = ''

	# - LIF neuron with plastic synapse ruled by LR1 (membrane changes for incoming spikes)
	elif plasticity_rule == 'LR1' and neuron_type == 'LIF':
		model_E_E = model_E_E_plastic
		pre_E_E = dict(Vepsp_transmission)
		pre_E_E = dict(xpre_update, **pre_E_E)
		pre_E_E = dict(rho_update_pre, **pre_E_E)
		pre_E_E = dict(w_update, **pre_E_E)
		post_E_E = post_E_E_LR1

	# - LIF neuron with plastic synapse ruled by LR1 (membrane does not change for incoming spikes)
	elif plasticity_rule == 'LR1' and (neuron_type == 'spikegenerator' or neuron_type == 'poisson'):
		model_E_E = model_E_E_plastic
		pre_E_E = dict(xpre_update)
		pre_E_E = dict(rho_update_pre, **pre_E_E)
		pre_E_E = dict(w_update, **pre_E_E)
		post_E_E = post_E_E_LR1

	# - LIF neuron with plastic synapse ruled by LR2 (membrane changes for incoming spikes)
	elif plasticity_rule == 'LR2' and neuron_type == 'LIF':
		model_E_E = model_E_E_plastic
		pre_E_E = dict(Vepsp_transmission)
		pre_E_E = dict(xpre_update, **pre_E_E)
		pre_E_E = dict(rho_update_pre, **pre_E_E)
		pre_E_E = dict(w_update, **pre_E_E)
		post_E_E = post_E_E_LR2

	# - LIF neuron with plastic synapse ruled by LR2 (membrane does not change for incoming spikes)
	elif plasticity_rule == 'LR2' and (neuron_type == 'spikegenerator' or neuron_type == 'poisson'):
		model_E_E = model_E_E_plastic
		pre_E_E = dict(xpre_update)
		pre_E_E = dict(rho_update_pre, **pre_E_E)
		pre_E_E = dict(w_update, **pre_E_E)
		post_E_E = post_E_E_LR2

	else:
		raise ValueError("invalid compination of learning rule and neuron type")

	return model_E_E, pre_E_E, post_E_E




