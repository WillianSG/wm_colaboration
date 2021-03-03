# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
"""
import setuptools
import os, sys, pickle, shutil
from brian2 import *
from numpy import *
from time import localtime

import matplotlib as mpl
import matplotlib.pyplot as plt

prefs.codegen.target = 'numpy'

helper_dir = 'helper_functions'
plotting_funcs_dir = 'plotting_functions'

# Parent directory
parent_dir = os.path.dirname(os.getcwd())

# Adding parent dir to list of dirs that the interpreter will search in
sys.path.append(os.path.join(parent_dir, helper_dir))
sys.path.append(os.path.join(parent_dir, plotting_funcs_dir))

# Helper modules
from att_net_obj import AttractorNetwork
from ext_attractors_plot import *
from learning_check_for_delay_activity import *

simulation_folder = os.path.join(parent_dir, 'net_simulation')

if not(os.path.isdir(simulation_folder)):
	os.mkdir(simulation_folder)

idt = localtime()  

simulation_id = str(idt.tm_year) + '{:02}'.format(idt.tm_mon) + \
	'{:02}'.format(idt.tm_mday) + '_'+ '{:02}'.format(idt.tm_hour) + '_' + \
	'{:02}'.format(idt.tm_min) + '_' + '{:02}'.format(idt.tm_sec)

# Starts a new scope for magic functions
start_scope()

# 1 ========== Execution parameters ==========

exp_name = '_learn_stimulus'

# results destination folder
path_sim_folder_superior = os.path.join(simulation_folder, simulation_id + exp_name)

if not(os.path.isdir(path_sim_folder_superior)):
	os.mkdir(path_sim_folder_superior)

nets = 1 # number of networks
sim_duration = 4*second # simulation time
w_max = 7.5*mV # max weight in EE connections

# STIMULUS pulse setttings
pulse_duration = 3*second # zero seconds not possible
pulse_duration_max = 1
pulse_duration_step = 1

# 2 ========== Initialize and store network ==========
print('\n> Setting up network...')

n = AttractorNetwork() # network class

n.t_run = sim_duration
n.id = simulation_id

n.int_meth_neur = 'linear'
n.int_meth_syn = 'euler'
n.stim_type_e = 'flat_to_E_fixed_size' # Stimulus type (square, circle, triangle, cross,...)

n.stim_size_e = 64
n.stim_offset_e = 80
n.stim_freq_e = 3900*Hz # 3900*Hz

n.stim_type_i = 'flat_to_I'
n.stim_size_i = n.N_i
n.stim_freq_i = 5000*Hz # 5000*Hz

# Spikemonitors
n.Input_to_E_mon_record = True
n.Input_to_I_mon_record = True
n.E_mon_record = True
n.I_mon_record = True

# @network_operation settings

# Rho snapshots
n.rho_matrix_snapshots = False
n.rho_matrix_snapshots_step = 1000*ms

# Weight snapshots
n.w_matrix_snapshots = False
n.w_matrix_snapshots_step = 1000*ms 

# Xpre snapshots
n.xpre_matrix_snapshots = False
n.xpre_matrix_snapshots_step = 1000*ms

# Xpost snapshots
n.xpost_matrix_snapshots = False
n.xpost_matrix_snapshots_step = 1000*ms

# variables for @network_operation function for stimulus change
n.stimulus_pulse = True
n.stimulus_pulse_clock_dt = pulse_duration

n.stim_freq_original = n.stim_freq_e

# Learning rule 
n.plasticity_rule = 'LR2' # LR1, LR2
n.parameter = '2.4' #
n.neuron_type = 'LIF' #
n.bistability = True

# Connection weights
n.w_input_e = 1*mV # 1
n.w_input_i = 1*mV # 1
n.w_e_e = 0.5*mV # 0.5
n.w_e_i = 1*mV # 1
n.w_i_e = 1*mV # 1
n.w_input_s = 67*mV # 1

# Other connection weight variables
n.w_e_e_max = w_max

n.fixed_attractor = False # [?]
n.fixed_attractor_wmax = 'all_max' # [?]

n.init_network_modules() # creates/initializes all objects/parameters/mons

# ======= Simulation

# Set pulse duration
n.stimulus_pulse_duration = n.stimulus_pulse_clock_dt

# Simulation flag of pulse duration
simulation_flag_pulse_duration = [0, n.stimulus_pulse_duration] # NO NEED

# Name of simulation
n.exp_type = exp_name + '_pulse_dur_'+ str(n.stimulus_pulse_clock_dt/second) + 's'

# Running simulation
print('\n> Simulating network...')

n.run_net()

# # =========================================================================

# delay_activity = learning_check_for_delay_activity(
# 	s_tpoints_input_e = n.Input_to_E_mon.t[:],
# 	n_inds_input_e = n.Input_to_E_mon.i[:],
# 	s_tpoints_e = n.E_mon.t[:],
# 	n_inds_e = n.E_mon.i[:], 
# 	t_run = n.t_run,
# 	stim_pulse_duration = n.stimulus_pulse_duration,
# 	size_attractor = len(n.stim_inds_original_E),
# 	plot_spiketrains = False,
# 	sim_id = n.id)

# # store network state if DA has been found
# if delay_activity[1]:
# 	print('\nDELAY ACTIVITY FOUND')

# 	n.net.store(
# 		name = 'network_' + simulation_id + '_final_state', 
# 		filename = os.path.join(
# 			path_sim_folder_superior,
# 			'network_' + simulation_id + '_final_state'
# 			)
# 		)

# 	print('\n  - re-simulating network...')

# 	n.stimulus_pulse_clock_dt = 1*second
# 	n.stimulus_pulse_duration = 1*second

# 	n.stim_freq_e = 0*Hz
# 	n.set_stimulus_e()

# 	# Name of simulation
# 	n.exp_type = exp_name + '_pulse_dur_'+ str(n.stimulus_pulse_clock_dt/second) + 's'

# 	n.run_net()

# 	# Input_to_E population 
# 	s_tpoints_input_e = n.Input_to_E_mon.t[:]
# 	n_inds_input_e = n.Input_to_E_mon.i[:]

# 	# Input_to_I population 
# 	s_tpoints_input_i = n.Input_to_I_mon.t[:]
# 	n_inds_input_i = n.Input_to_I_mon.i[:]

# 	# Excitatory population
# 	s_tpoints_e = n.E_mon.t[:]
# 	n_inds_e = n.E_mon.i[:]

# 	# Inhibitory population
# 	s_tpoints_i = n.I_mon.t[:]
# 	n_inds_i = n.I_mon.i[:]

# 	s_tpoints = [
# 		s_tpoints_input_e, 
# 		s_tpoints_input_i, 
# 		s_tpoints_e,
# 		s_tpoints_i
# 	]

# 	n_inds = [
# 		n_inds_input_e,
# 		n_inds_input_i,
# 		n_inds_e,
# 		n_inds_i
# 	]

# 	# Pickling
# 	path_sim = n.path_sim
# 	sim_id = n.id
# 	t_run = n.t_run
# 	exp_type = n.exp_type

# 	stim_type_e = n.stim_type_e
# 	stim_size_e = n.stim_size_e
# 	stim_freq_e = n.stim_freq_e
# 	stim_offset_e = n.stim_offset_e
# 	stim_type_i = n.stim_type_i
# 	stim_size_i = n.stim_size_i
# 	stim_freq_i = n.stim_freq_i
# 	stim_pulse_duration = n.stimulus_pulse_duration

# 	N_input_e = n.N_input_e
# 	N_input_i = n.N_input_i

# 	N_e = n.N_e
# 	N_i = n.N_i

# 	len_stim_inds_original_E = len(n.stim_inds_original_E)
# 	w_e_e_max = n.w_e_e_max
# 	rho_matrix_snapshots_step = n.rho_matrix_snapshots_step

# 	folder_snaps = os.path.split(n.path_snapshots)[1]
# 	path_rho = n.path_snapshots

# 	w_matrix_snapshots_step = n.w_matrix_snapshots_step
# 	path_w = n.path_snapshots

# 	xpre_matrix_snapshots_step = n.xpre_matrix_snapshots_step
# 	path_xpre = n.path_snapshots
# 	xpost_matrix_snapshots_step = n.xpost_matrix_snapshots_step

# 	path_xpost = n.path_snapshots
# 	plasticity_rule = n.plasticity_rule

# 	# =====================================================================

# 	fn = os.path.join(n.path_sim, n.id +'_' + n.exp_type + '.pickle')

# 	with open(fn, 'wb') as f:
# 		pickle.dump((
# 			path_sim, 
# 			sim_id, 
# 			nets,
# 			simulation_flag_pulse_duration,
# 			t_run, 
# 			exp_type,
# 			stim_type_e, 
# 			stim_type_e,
# 			stim_size_e,
# 			stim_freq_e,
# 			stim_offset_e,
# 			stim_type_i,
# 			stim_size_i,
# 			stim_freq_i,
# 			stim_pulse_duration,
# 			[],
# 			N_input_e, 
# 			N_input_i, 
# 			N_e, 
# 			N_i,
# 			len_stim_inds_original_E,
# 			w_e_e_max, 
# 			rho_matrix_snapshots_step,
# 			folder_snaps,
# 			w_matrix_snapshots_step, 
# 			path_w,
# 			xpre_matrix_snapshots_step, 
# 			path_xpre,
# 			xpost_matrix_snapshots_step, 
# 			path_xpost,
# 			s_tpoints_input_e, 
# 			s_tpoints_input_i, 
# 			s_tpoints_e, 
# 			s_tpoints_i,
# 			n_inds_input_e,
# 			n_inds_input_i,
# 			n_inds_e, 
# 			n_inds_i), f)

# 	print('\nSimulation data pickled to ', fn)

# 	# Move current simulation folder to superior simulation folder
# 	shutil.move(n.path_sim, path_sim_folder_superior)
# else:
# 	print('\nNO delay activity')

# ext_attractors_plot(
# 	path_sim_folder_superior, 
# 	sim_id, 
# 	exp_type, 
# 	spiketrains_and_histograms = True,
# 	rho_matrix_snapshots = False, 
# 	w_matrix_snapshots = False, 
# 	xpre_matrix_snapshots = False, 
# 	xpost_matrix_snapshots = False,
# 	learning_performance = False, 
# 	check_wmatrices = False)

print('\nattractor_stability.py - END.\n')