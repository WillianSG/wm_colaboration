# -*- coding: utf-8 -*-
"""
@author: wgirao

Input:

Output:

Comments:

Loaded data:
- 0 w_init:
- 1 sim_rep:
- 2 resul_per_pre_rate:
- 3 f_pre:
- 4 f_pos:
- 5 min_freq:
- 6 max_freq:
- 7 step:
- 8 exp_type:
- 9 plasticity_rule:
- 10 parameter_set:
- 11 bistability:
- 12 dt_resolution:
- 13 t_run:
- 14 int_meth_syn:
"""
import os, sys, pickle
from numpy import *
import matplotlib.pyplot as plt

ltd_exp_data = pickle.load(
	open('stdp_trans_probabi__1.0_2.2_True__last_rho.pickle', "rb" ))

fig = plt.figure()
ax = plt.subplot(111)

for x in range(0, len(ltd_exp_data[2])):
	ax.plot(
		ltd_exp_data[4], ltd_exp_data[2][x], 
		label = str((x+1)*5) + 'Hz',
		linestyle = '--', 
		marker = '.')

plt.title('LTD Transition Probability')
plt.xlabel('Postsynaptic frequency (Hz)')
plt.ylabel('P')

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(
	loc = 'center left', 
	bbox_to_anchor = (1, 0.5), 
	title = 'pre- rate',
	fontsize = 8)

plt.savefig(str(ltd_exp_data[8]) + \
	'_' + str(ltd_exp_data[9]) + \
	'_' + str(ltd_exp_data[10]) + \
	'_' + str(ltd_exp_data[11]) + \
	'_' + str(ltd_exp_data[13]) + \
	'.png')

print('\nplot_transition_probab.py - END.\n')