# -*- coding: utf-8 -*-
"""
@author: wgirao
@based-on: asonntag
"""

def load_rule_params(plasticity_rule, parameter_set):
	from brian2 import ms, mV
	xpre_jump = 1 # jump of x_pre
	xpost_jump = 1 # jump of x_post
	rho_init = 0.5 # initial rho value
	tau_rho = 350000*ms # rho time constant
	rho_min = 0.0 # DOWN state
	rho_max = 1 # UP state
	thr_b_rho = 0.5 # bistability threshold
	alpha = 1 # slope of bistability
	beta = alpha

	if plasticity_rule == 'LR2':
		if parameter_set =='2.1':
			tau_xpre = 13*ms # time constant x_pre 
			tau_xpost = 33*ms # time constant x_post
			xpre_factor = 0.1 # scaling factor positive efficacy change
			thr_post = 0.4 # threshold for x_post
			thr_pre = 0.2 # threshold for x_pre
			rho_dep = -0.05 # negative efficacy change
			rho_dep2 = rho_dep # additional negative efficacy change 
		elif parameter_set =='2.2':
			tau_xpre = 13*ms # time constant x_pre 
			tau_xpost = 33*ms # time constant x_post
			xpre_factor = 0.013 # scaling factor positive efficacy change
			thr_post = 0.4 #0.4# threshold for x_post
			thr_pre = 0.5 # threshold for x_pre
			rho_dep = -0.0008 # negative efficacy change
			rho_dep2 = rho_dep*10 # additional negative efficacy change 
		elif parameter_set =='2.3':
			tau_xpre = 13*ms # time constant x_pre 
			tau_xpost = 33*ms # time constant x_post
			xpre_factor = 0.017 # scaling factor positive efficacy change
			thr_post = 0.4 #0.4# threshold for x_post
			thr_pre = 0.5 # threshold for x_pre
			rho_dep = -0.00055 # negative efficacy change
			rho_dep2 = rho_dep*10 # additional negative efficacy change 
		elif parameter_set =='2.4':
			tau_xpre = 13*ms # time constant x_pre 
			tau_xpost = 33*ms # time constant x_post
			xpre_factor = 0.21 # scaling factor positive efficacy change
			thr_post = 0.4 #0.4# threshold for x_post
			thr_pre = 0.5 # threshold for x_pre
			rho_dep = -0.008 # negative efficacy change
			rho_dep2 = rho_dep*10 # additional negative efficacy change
	else: # default '2.1'
		tau_xpre = 13*ms
		tau_xpost = 33*ms
		xpre_factor = 0.1
		thr_post = 0.4
		thr_pre = 0.2
		rho_dep = -0.05
		rho_dep2 = rho_dep

	w_max = 1*mV

	return tau_xpre,\
	tau_xpost,\
	xpre_jump,\
	xpost_jump,\
	rho_dep,\
	rho_dep2,\
	rho_init,\
	tau_rho,\
	thr_post,\
	thr_pre,\
	thr_b_rho,\
	rho_min,\
	rho_max,\
	alpha,\
	beta,\
	xpre_factor,\
	w_max