# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System

Function:
- Creates the differential equations used to describe the synaptic model of weight (w) update as argument for Brian2's Synapses() method.

Script arguments:
-

Script output:
-

Comments:
- As the weight (in units of volts) updated depends on other admentional variables, 'rho' is used as a dimensionless variable that helps to calculate a weight update in volts.

Input:
- plasticity_rule (str): rule's name
- neuron_type (str): spiking activity
- bistability (bool): with or without it

Output:
- model_E_E: synaptic model equation
- pre_E_E: on presy- spike
- post_E_E: on postsy- spike
"""


def load_synapse_model(plasticity_rule, neuron_type, bistability, stoplearning=False, resources=False):
    # Non-plastic synapse
    model_E_E_non_plastic = '''w : volt'''

    # Plastic synapse (STDP)
    if bistability == True:
        if stoplearning:
            model_E_E_plastic = ''' 
			w : volt
			plastic : boolean (shared)
			dxpre/dt = (-xpre / tau_xpre)*int(plastic) : 1 (clock-driven)
			dxpost/dt = (-xpost / tau_xpost)*int(plastic) : 1 (clock-driven)
			dxstop/dt = (-xstop / tau_xstop)*int(plastic) : 1 (clock-driven) 
			drho/dt = (int(rho > thr_b_rho)*int(rho < rho_max)  * alpha*int(plastic) -
			int(rho <= thr_b_rho) * beta*int(plastic) * int(rho > rho_min)) / tau_rho 
				: 1  (clock-driven)'''
        elif resources and plasticity_rule == 'LR4':
            model_E_E_plastic = ''' 
			w : volt
			plastic : boolean (shared)
            plastic_2 : boolean (shared)
			du/dt = ((U - u)/tau_f)*int(plastic_2) : 1 (clock-driven)
			dx_/dt = ((1 - x_)/tau_d)*int(plastic_2) : 1 (clock-driven)
			dxpre/dt = (-xpre / tau_xpre)*int(plastic) : 1 (clock-driven)
			dxpost/dt = (-xpost / tau_xpost)*int(plastic) : 1 (clock-driven)
			drho/dt = (int(rho > thr_b_rho)*int(rho < rho_max)  * alpha*int(plastic) -
			int(rho <= thr_b_rho) * beta*int(plastic) * int(rho > rho_min)) / tau_rho 
				: 1  (clock-driven)'''
        else:
            model_E_E_plastic = ''' 
			w : volt
			plastic : boolean (shared)
			dxpre/dt = (-xpre / tau_xpre)*int(plastic) : 1 (clock-driven)
			dxpost/dt = (-xpost / tau_xpost)*int(plastic) : 1 (clock-driven)
			drho/dt = (int(rho > thr_b_rho)*int(rho < rho_max)  * alpha*int(plastic) -
			int(rho <= thr_b_rho) * beta*int(plastic) * int(rho > rho_min)) / tau_rho 
				: 1  (clock-driven)'''
    elif bistability == False:
        if stoplearning:
            model_E_E_plastic = ''' 
			w : volt
			rho : 1
			plastic : boolean (shared)
			dxpre/dt = (-xpre / tau_xpre)*int(plastic) : 1 (clock-driven)
			dxpost/dt = (-xpost / tau_xpost)*int(plastic) : 1 (clock-driven)
			dxstop/dt = (-xstop / tau_xstop)*int(plastic) : 1 (clock-driven)'''
        else:
            model_E_E_plastic = ''' 
			w : volt
			rho : 1
			plastic : boolean (shared)
			dxpre/dt = (-xpre / tau_xpre)*int(plastic) : 1 (clock-driven)
			dxpost/dt = (-xpost / tau_xpost)*int(plastic) : 1 (clock-driven)'''
    else:
        print('Bistabilty setting unclear')

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

    # - On post spike (LR3)
    """
    xpost_jump: A_post
    xpre_factor: c
    rho_dep2: rho_neg2
    """
    if stoplearning:
        post_E_E_LR3 = '''xstop = xstop + xstop_jump * (xstop_max - xstop) * int(plastic)
		xpost = clip((xpost + xpost_jump * int(plastic)), 0.0, 1.0)
		rho = clip((rho + ((xpre * xpre_factor + rho_neg2 *int(xpre < thr_pre) * int(xpre > 0))*int(plastic))*int(xstop < thr_stop_h)*int(xstop > thr_stop_l)), rho_min, rho_max)
		w = w_max*int(rho > 0.5) + 0*mV'''
    else:
        post_E_E_LR3 = '''xpost = clip((xpost + xpost_jump * int(plastic)), 0.0, 1.0)
			rho = clip((rho + (xpre * xpre_factor + rho_neg2 *int(xpre < thr_pre)*int(xpre > 0))*int(plastic)), rho_min, rho_max)
			w = w_max*int(rho > 0.5) + 0*mV'''

    # - On post spike (LR4)
    post_E_E_LR4 = '''xpost = clip((xpost + xpost_jump * int(plastic)), 0.0, 1.0)
		rho = clip((rho + (xpre * xpre_factor + rho_neg2 *int(xpre < thr_pre)*int(xpre > 0))*int(plastic)), rho_min, rho_max)
		w = w_max*rho'''

    # - On pre (1) spike (LR3)
    xpre_update_LR3 = {'xpre_update': '''xpre = xpre + xpre_jump * (xpre_max - xpre) * int(plastic)'''}

    # - On pre (1) spike (LR4)
    xpre_update_LR4 = {'xpre_update': '''xpre = xpre + xpre_jump * (xpre_max - xpre) * int(plastic)'''}
    x_update_LR4 = {'x_update': '''x_ = x_ - u*x_*int(plastic_2)'''}
    u_update_LR4 = {'u_update': '''u = u + U*(1 - u)*int(plastic_2)'''}

    # - On pre (2) spike (both LR1/LR2/LR3)
    """
    xpre_jump: A_pre
    rho_dep: rho_neg
    """
    xpre_update = {'xpre_update': '''xpre = xpre + xpre_jump * int(plastic)'''}

    # weight update on pre spike
    w_update = {'w_update': ''' w = w_max*int(rho > 0.5) + 0*mV'''}

    w_update_LR4 = {'w_update': ''' w = w_max*rho'''}

    if stoplearning:
        rho_update_pre = {
            'rho_update_pre': '''rho = clip(rho + (rho_neg *int(xpost > thr_post)*int(xstop < thr_stop_h)*int(plastic))*int(xstop > thr_stop_l), rho_min, rho_max)'''}
    else:
        rho_update_pre = {
            'rho_update_pre': '''rho = clip(rho + (rho_neg *int(xpost > thr_post)*int(plastic)), rho_min, rho_max)'''}

    # Defines the argument 'on_pre' for Brian2 - same as on_pre='v_post += w'
    """
    Vepsp: effect of presynaptic activity on postsynaptic mem. potential.
    """
    Vepsp_transmission = {'Vepsp_transmission': '''Vepsp += w'''}

    Vepsp_transmission_LR4 = {'Vepsp_transmission': '''Vepsp += w*(x_*u)'''}
    # Vepsp_transmission_LR4 = {'Vepsp_transmission': '''Vepsp += w'''}

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

    elif plasticity_rule == 'LR3' and (neuron_type == 'spikegenerator' or neuron_type == 'poisson'):
        model_E_E = model_E_E_plastic
        pre_E_E = dict(xpre_update_LR3)
        pre_E_E = dict(rho_update_pre, **pre_E_E)
        pre_E_E = dict(w_update, **pre_E_E)
        post_E_E = post_E_E_LR3

    elif plasticity_rule == 'LR3' and neuron_type == 'LIF':
        model_E_E = model_E_E_plastic
        pre_E_E = dict(Vepsp_transmission)
        pre_E_E = dict(xpre_update_LR3, **pre_E_E)
        pre_E_E = dict(rho_update_pre, **pre_E_E)
        pre_E_E = dict(w_update, **pre_E_E)
        post_E_E = post_E_E_LR3

    elif plasticity_rule == 'LR4' and neuron_type == 'LIF':
        model_E_E = model_E_E_plastic
        pre_E_E = dict(Vepsp_transmission_LR4)
        pre_E_E = dict(xpre_update_LR4, **pre_E_E)
        pre_E_E = dict(x_update_LR4, **pre_E_E)
        pre_E_E = dict(u_update_LR4, **pre_E_E)
        pre_E_E = dict(rho_update_pre, **pre_E_E)
        pre_E_E = dict(w_update_LR4, **pre_E_E)
        post_E_E = post_E_E_LR4

    elif plasticity_rule == 'LR4' and (neuron_type == 'spikegenerator' or neuron_type == 'poisson'):
        model_E_E = model_E_E_plastic
        pre_E_E = dict(xpre_update_LR4)
        pre_E_E = dict(x_update_LR4, **pre_E_E)
        pre_E_E = dict(u_update_LR4, **pre_E_E)
        pre_E_E = dict(rho_update_pre, **pre_E_E)
        pre_E_E = dict(w_update_LR4, **pre_E_E)
        post_E_E = post_E_E_LR4

    else:
        raise ValueError("invalid compination of learning rule and neuron type")

    return model_E_E, pre_E_E, post_E_E
