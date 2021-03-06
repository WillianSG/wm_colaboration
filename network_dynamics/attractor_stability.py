# -*- coding: utf-8 -*-
"""
@author: wgirao

Comments:
- Learning of stimulus in an attractor network.
- https://brian2.readthedocs.io/en/stable/user/running.html
- Plotting scripts consume the pickled data

-------- TO DO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
- Use "net = Network(collect())"
- Use to switch plasticity on or off S = Synapses(..., '''...
                     plastic : boolean (shared)
                     ...''')
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
from att_net_obj_mdf import AttractorNetwork
from control_plot_learned_attractor import *

# 1 ====================== Results/experiment folders =====================
network_sim_dir = os.path.join(
	parent_dir, 
	'network_results')

if not(os.path.isdir(network_sim_dir)):
	os.mkdir(network_sim_dir)

script_results_path = os.path.join(
	network_sim_dir,
	'attractor_stability')

if not(os.path.isdir(script_results_path)):
	os.mkdir(script_results_path)

idt = localtime()  

simulation_id = str(idt.tm_year) + '{:02}'.format(idt.tm_mon) + \
	'{:02}'.format(idt.tm_mday) + '_'+ '{:02}'.format(idt.tm_hour) + '_' + \
	'{:02}'.format(idt.tm_min) + '_' + '{:02}'.format(idt.tm_sec)

# Starts a new scope for magic functions
# start_scope()

exp_name = '_attractor_stability'

# results destination folder
simulation_results_path = os.path.join(script_results_path, simulation_id + exp_name)

if not(os.path.isdir(simulation_results_path)):
	os.mkdir(simulation_results_path)

# 2 ====================== Simulation parameters ==========================
print('\n> initializing network')

total_simulated_time = 0*second

t_run = 5*second
stimulus_pulse_clock_dt = 5*second

t_run_silencing = 1*second
stimulus_pulse_clock_dt_silencing = 1*second

nets = 1 				# number of networks

n = AttractorNetwork() 	# network class
n.simulation_results_path = simulation_results_path

# setting sim. timerun for learning
n.t_run = t_run 		# simulation time
n.id = simulation_id

# variables for @network_operation function for stimulus change
n.stimulus_pulse = False
n.stimulus_pulse_clock_dt = stimulus_pulse_clock_dt # zero seconds not possible
n.stimulus_pulse_duration = n.stimulus_pulse_clock_dt # Set pulse duration

n.int_meth_neur = 'linear'
n.int_meth_syn = 'euler'
n.stim_type_e = 'flat_to_E_fixed_size' # square, circle, triangle, cross,..

n.stim_size_e = 64
n.stim_offset_e = 0
n.stim_freq_e = 3900*Hz 	# 3900*Hz

n.stim_type_i = 'flat_to_I'
n.stim_size_i = n.N_i
n.stim_freq_i = 5000*Hz 	# 5000*Hz

# Spikemonitors
n.Input_to_E_mon_record = True
n.Input_to_I_mon_record = True
n.E_mon_record = True
n.I_mon_record = True

n.stim_freq_original = n.stim_freq_e

# Learning rule 
n.plasticity_rule = 'LR2' 	# LR1, LR2
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
n.w_e_e_max = 7.5*mV 		# max weight in EE connections

n.fixed_attractor = False # [?]
n.fixed_attractor_wmax = 'all_max' # [?]

n.init_network_modules() # creates/initializes all objects/parameters/mons

# 2.1 saving net initialized state ========================================
n.net.store(
	name = 'initialized_net_' + simulation_id, 
	filename = os.path.join(
		simulation_results_path,
		'initialized_net_' + simulation_id
		)
)

# 3 ====================== learning cue ===================================
n.exp_type = exp_name + '_learning_cue'

print('\n> forming attractor for cue stimulus\n')

n.save_monitors() 			# monitors for current sim

n.run_network(period = 2) 	# Running simulation

total_simulated_time += n.t_run

# 3.1 saving net trained on cue state =====================================
n.net.store(
	name = 'trained_cue_net_' + simulation_id, 
	filename = os.path.join(
		simulation_results_path,
		'trained_cue_net_' + simulation_id
		)
)

# 3.2 - pickling and plotting =============================================
fn = os.path.join(
	simulation_results_path,
	n.id +'_' + n.exp_type + '_trained.pickle')

spk_mons = n.get_stimulus_monitors() # [input_to_E, input_to_I, E_mon, I_mon]

s_tpoints_input_e = spk_mons[0].t[:]
n_inds_input_e = spk_mons[0].i[:]

s_tpoints_input_i = spk_mons[1].t[:]
n_inds_input_i = spk_mons[1].i[:]

s_tpoints_e = spk_mons[2].t[:]
n_inds_e = spk_mons[2].i[:]

s_tpoints_i = spk_mons[3].t[:]
n_inds_i = spk_mons[3].i[:]

with open(fn, 'wb') as f:
	pickle.dump((
		n.plasticity_rule,
		nets,
		n.path_sim, 
		n.id,
		n.t_run, 
		n.exp_type,
		n.stim_type_e,
		n.stim_size_e,
		n.stim_freq_e,
		n.stim_offset_e,
		n.stim_type_i,
		n.stim_size_i,
		n.stim_freq_i,
		n.stimulus_pulse_duration,
		n.N_input_e, 
		n.N_input_i, 
		n.N_e, 
		n.N_i,
		len(n.stim_inds_original_E),
		n.w_e_e_max,
		s_tpoints_input_e,
		n_inds_input_e,
		s_tpoints_input_i,
		n_inds_input_i,
		s_tpoints_e,
		n_inds_e,
		s_tpoints_i,
		n_inds_i), f)

print('\n -> pickled to: ', fn)

control_plot_learned_attractor(
	network_state_path = simulation_results_path,
	pickled_data = n.id +'_' + n.exp_type,
	name = '_trained.pickle')

# 4 ====================== learning followup stimuli ======================
stimuli = ['square', 'circle', 'triangle', 'cross']
# stimuli = ['flat_to_E_fixed_size', 'flat_to_E_fixed_size']

t_run_count_aux = 0

for stimulus in stimuli:
	t_run_count_aux += 1
	# 4.1 silencing current E activity ====================================
	print('\n -> silencing activity\n')

	# setting sim. timerun for silencing
	n.t_run = t_run_silencing
	n.stimulus_pulse_clock_dt = stimulus_pulse_clock_dt_silencing
	n.stimulus_pulse_duration = n.stimulus_pulse_clock_dt

	n.silence_activity()			# silencing current E activity
	n.run_network(report = None)	# simulate

	total_simulated_time += n.t_run # accounting for simulated time

	n.resume_silencing_activity()	# resuming silencing population

	# 4.2 learning followup stimuli =======================================

	# setting sim. timerun for learning
	n.t_run = t_run
	n.stimulus_pulse_clock_dt = stimulus_pulse_clock_dt
	n.stimulus_pulse_duration = n.stimulus_pulse_clock_dt

	# n.change_stimulus_e(stimulus = stimulus, offset = n.stim_offset_e + 42)
	n.change_stimulus_e(stimulus = stimulus)

	n.exp_type = exp_name + '_learning_' + n.stim_type_e

	print('\n> learning stimulus [ ', n.stim_type_e, ' ]\n')

	# n.save_monitors(opt = 'skewed_' + str(n.stim_offset_e))
	n.save_monitors()

	n.run_network(period = 2)

	# 4.3 saving net trained on cue state =================================
	n.net.store(
		name = 'trained_' + n.stim_type_e + '_' + str(n.stim_offset_e) + '_net_' + simulation_id, 
		filename = os.path.join(
			simulation_results_path,
			'trained_' + n.stim_type_e + '_' + str(n.stim_offset_e) + '_net_' + simulation_id
			)
	)

	# 4.4 - pickling and plotting =========================================

	fn = os.path.join(
		simulation_results_path,
		n.id +'_' + n.exp_type + '_trained.pickle')

	spk_mons = n.get_stimulus_monitors()

	s_tpoints_input_e = [x - total_simulated_time for x in spk_mons[0].t[:]]
	n_inds_input_e = spk_mons[0].i[:]

	s_tpoints_input_i = [x - total_simulated_time for x in spk_mons[1].t[:]]
	n_inds_input_i = spk_mons[1].i[:]

	s_tpoints_e = [x - total_simulated_time for x in spk_mons[2].t[:]]
	n_inds_e = spk_mons[2].i[:]

	s_tpoints_i = [x - total_simulated_time for x in spk_mons[3].t[:]]
	n_inds_i = spk_mons[3].i[:]

	total_simulated_time += n.t_run # accounting for simulated time

	with open(fn, 'wb') as f:
		pickle.dump((
			n.plasticity_rule,
			nets,
			n.path_sim, 
			n.id,
			n.t_run, 
			n.exp_type,
			n.stim_type_e,
			n.stim_size_e,
			n.stim_freq_e,
			n.stim_offset_e,
			n.stim_type_i,
			n.stim_size_i,
			n.stim_freq_i,
			n.stimulus_pulse_duration,
			n.N_input_e, 
			n.N_input_i, 
			n.N_e, 
			n.N_i,
			len(n.stim_inds_original_E),
			n.w_e_e_max,
			s_tpoints_input_e,
			n_inds_input_e,
			s_tpoints_input_i,
			n_inds_input_i,
			s_tpoints_e,
			n_inds_e,
			s_tpoints_i,
			n_inds_i), f)

	print('\n -> pickled to: ', fn)

	control_plot_learned_attractor(
		network_state_path = simulation_results_path,
		pickled_data = n.id +'_' + n.exp_type,
		name = '_trained.pickle',
		opt = str(n.stim_offset_e))

print('\nattractor_stability.py - END.\n')