# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System

Function:
- 

Script arguments:
-

Script output:
-

Comments:
-
"""

from brian2 import ms, mV

def load_rule_parameters(
	plasticity_rule, 
	parameter_set, 
	efficacy_init = 0.5, 
	max_w = 7.5*mV):

	tau_xpre = 0.0*ms
	tau_xpost = 0.0*ms
	xpre_jump = 1
	xpost_jump = 1
	rho_neg = 0.0
	rho_neg2 = 0.0
	rho_init = efficacy_init
	tau_rho = 350000*ms
	thr_post = 0.0
	thr_pre = 0.0
	thr_b_rho = 0.5
	rho_min = 0.0
	rho_max = 1.0
	alpha = 1.0
	beta = alpha
	xpre_factor = 0.0
	w_max = max_w
	xpre_min = 0.0
	xpost_min = 0.0
	xpost_max = 1.0
	xpre_max = 1.0
	tau_xstop = 0.0*ms
	xstop_jump = 0.0
	xstop_max = 0.0
	xstop_min = 0.0
	thr_stop_h = 0.0
	thr_stop_l = 0.0
	U = 0.0
	tau_d = 0*ms
	tau_f = 0*ms

	if plasticity_rule == 'LR2':
		if parameter_set =='2.4':
			tau_xpre = 13*ms
			tau_xpost = 33*ms
			xpre_factor = 0.21
			thr_post = 0.4
			thr_pre = 0.5
			rho_neg = -0.008
			rho_neg2 = rho_neg*10
	elif plasticity_rule == 'LR3':
		if parameter_set == '1.X1':
			tau_xpre = 30*ms
			tau_xpost = 45*ms 
			tau_rho = 40000*ms
			xpre_jump = 0.4
			xpost_jump = 0.5
			thr_post = 0.33
			thr_pre = 0.15
			rho_neg = -0.021
			rho_neg2 = -0.002
			xpre_factor = 0.06
			w_max = 5*mV
			tau_xstop = 600*ms
			xstop_jump = 0.03
			xstop_max = 1.0
			xstop_min = 0.0
			thr_stop_h = 0.59
			thr_stop_l = 0.1
		elif parameter_set == '2.0':
			tau_xpre = 20*ms
			tau_xpost = 35*ms 
			tau_rho = 20000*ms
			xpre_jump = 0.4
			xpost_jump = 0.5
			thr_post = 0.33
			thr_pre = 0.25
			rho_neg = -0.021
			rho_neg2 = -0.002
			xpre_factor = 0.1
			w_max = 7.5*mV
			tau_xstop = 600*ms
			xstop_jump = 0.03
			xstop_max = 1.0
			xstop_min = 0.0
			thr_stop_h = 0.59
			thr_stop_l = 0.1
		elif parameter_set == '3.0':
			tau_xpre = 25*ms
			tau_xpost = 35*ms 
			tau_rho = 20000*ms
			xpre_jump = 0.4
			xpost_jump = 0.5
			thr_post = 0.33
			thr_pre = 0.1
			rho_neg = -0.021
			rho_neg2 = -0.002
			xpre_factor = 0.2
			w_max = 5*mV
	elif plasticity_rule == 'LR4':
		if parameter_set == '1.0':
			tau_xpre = 20*ms
			tau_xpost = 35*ms 
			tau_rho = 20000*ms
			xpre_jump = 0.4
			xpost_jump = 0.5
			thr_post = 0.33
			thr_pre = 0.25
			rho_neg = -0.021
			rho_neg2 = -0.002
			xpre_factor = 0.1
			w_max = 2*mV
			U = 0.05
			tau_d = 200*ms
			tau_f = 2000*ms
		elif parameter_set == '2.0':
			tau_xpre = 25*ms
			tau_xpost = 35*ms 
			tau_rho = 20000*ms
			xpre_jump = 0.4
			xpost_jump = 0.5
			thr_post = 0.33
			thr_pre = 0.1
			rho_neg = -0.021
			rho_neg2 = -0.002
			xpre_factor = 0.2
			w_max = 5*mV
			U = 0.2
			tau_d = 200*ms
			tau_f = 900*ms
	else: # default '2.4'
		tau_xpre = 13*ms
		tau_xpost = 33*ms
		xpre_factor = 0.21
		thr_post = 0.4
		thr_pre = 0.5
		rho_neg = -0.008
		rho_neg2 = rho_neg*10

	return tau_xpre,\
	tau_xpost,\
	xpre_jump,\
	xpost_jump,\
	rho_neg,\
	rho_neg2,\
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
	w_max,\
	xpre_min,\
	xpost_min,\
	xpost_max,\
	xpre_max,\
	tau_xstop,\
	xstop_jump,\
	xstop_max,\
	xstop_min,\
	thr_stop_h,\
	thr_stop_l,\
	U,\
	tau_d,\
	tau_f