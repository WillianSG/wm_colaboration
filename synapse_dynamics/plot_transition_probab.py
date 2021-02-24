# -*- coding: utf-8 -*-
"""
@author: wgirao

Input:

Output:

Comments:

Loaded data:
- w_init:
- sim_loop_rep:
- transition_probabilities:
- f_pre:
- f_pos:
- min_freq:
- max_freq:
- step:
- exp_type:
- plasticity_rule:
- parameter_set:
- bistability:
- dt_resolution:
- t_run:
- noise:
- int_meth_syn:
"""
import os, sys, pickle
from numpy import *
import matplotlib.pyplot as plt


ltp_exp_data = pickle.load(
	open('stdp_trans_probabi__0.0_w_final.pickle', "rb" ))

ltd_exp_data = pickle.load(
	open('stdp_trans_probabi__1.0_w_final.pickle', "rb" ))

plt.plot(ltd_exp_data[4], ltd_exp_data[2], label = 'LTD')
plt.plot(ltd_exp_data[4], ltp_exp_data[2], label = 'LTP')

plt.title('Transition Probabilities')
plt.xlabel('Postsynaptic frequency (Hz)')
plt.ylabel('Probability')

plt.legend()

plt.savefig(str(ltd_exp_data[8]) + \
	'_' + str(ltd_exp_data[9]) + \
	'_' + str(ltd_exp_data[10]) + \
	'_' + str(ltd_exp_data[11]) + \
	'.png')

print('\nplot_transition_probab.py - END.\n')