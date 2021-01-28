# -*- coding: utf-8 -*-
"""
@author: wgirao
@based-on: asonntag
@original: adapted from Lehfeldt

Comments:
- Learning of stimulus in an attractor network with different stimulus durations.
- Ext_attractors is to look at varying stimulus duration for a given maximum weight.
"""
import setuptools
import os, sys, pickle, shutil
from brian2 import *
from numpy import *
from time import localtime

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

exp_name = '_learning_att'

# results destination folder
path_sim_folder_superior = os.path.join(simulation_folder, simulation_id + exp_name)
if not(os.path.isdir(path_sim_folder_superior)):
	os.mkdir(path_sim_folder_superior)

nets = 1 # number of networks ('?')
sim_duration = 10*second # simulation time
w_max = 7.25*mV # max weight in EE connections

# Stimulus pulse setttings (?)
pulse_duration_min = 3 # zero seconds not possible
pulse_duration_max = 3
pulse_duration_step = 1
pulse_durations = arange(pulse_duration_min, pulse_duration_max+pulse_duration_step, pulse_duration_step)*second

# 2 ========== Initialize and store network (?) ==========

for i in arange(0, nets, 1):
	print('Simulating network ', i, ' of ', nets)

	n = AttractorNetwork() # network class

	n.t_run = sim_duration
	n.id = simulation_id

	n.int_meth_neur = 'linear'
	n.int_meth_syn = 'euler'
	n.stim_type_e = 'flat_to_E_fixed_size'

	n.stim_size_e = 64
	n.stim_freq_e = 3900*Hz
	n.stim_offset_e = 96

	n.stim_type_i = 'flat_to_I'

	n.stim_size_i = n.N_i
	n.stim_freq_i = 5000*Hz

	# Spikemonitors
	n.Input_to_E_mon_record = True
	n.Input_to_I_mon_record = True
	n.E_mon_record = True
	n.I_mon_record = True

	# @network_operation settings

	# Rho snapshots
	n.rho_matrix_snapshots = False
	n.rho_matrix_snapshots_step = 1000 *ms

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
	n.stimulus_pulse_clock_dt = min(pulse_durations)

	n.stim_freq_original = n.stim_freq_e

	# Learning rule 
	n.plasticity_rule = 'LR2' # LR1, LR2
	n.parameter = '2.4' #
	n.neuron_type = 'LIF' #
	n.bistability = True

	# Connection weights
	n.w_input_e = 1*mV
	n.w_input_i = 1*mV
	n.w_e_e = 0.5*mV
	n.w_e_i = 1*mV        
	n.w_i_e = 1*mV

	# Other connection weight variables
	n.w_e_e_max = w_max

	n.fixed_attractor = False # [?]
	n.fixed_attractor_wmax = 'all_max' # [?]

	n.init_network_modules() # creates/initializes all objects/parameters/mons

	n.net.store(
		name = 'network_initial_state', 
		filename = os.path.join(
			path_sim_folder_superior,
			'network_initial_state'
			)
	) # [?] - net.store (from network obj in Brian2)

	# ======= Loop : pulse duration
	for d in np.arange(0, len(pulse_durations), 1):
		# Restore network state with loaded learning rule parameters
		n.net.restore(name = 'network_initial_state', filename = os.path.join(path_sim_folder_superior, 'network_initial_state'))

		# Set pulse duration
		n.stimulus_pulse_duration = pulse_durations[d]

		# Simulation flag of pulse duration - [?]
		simulation_flag_pulse_duration = [d, n.stimulus_pulse_duration]

		# Name of simulation
		n.exp_type = 'learn_network_' + str(i+1)  + '_pulse_dur_'+ str(pulse_durations[d]/second) + 's'

		# Running simulation
		print('\nLearning... [net ', i+1 , ' | pulse duration', str(pulse_durations[d]) + ']')

		n.run_net()

		print('Finished learning: net ', i+1)

		# Simulation data storage

		# 1) Timepoints of spikes (s_tpoints) and neuron indices (n_inds)

		# Input_to_E population 
		s_tpoints_input_e = n.Input_to_E_mon.t[:]
		n_inds_input_e = n.Input_to_E_mon.i[:]

		# Input_to_I population 
		s_tpoints_input_i = n.Input_to_I_mon.t[:]
		n_inds_input_i = n.Input_to_I_mon.i[:]

		# Excitatory population
		s_tpoints_e = n.E_mon.t[:]
		n_inds_e = n.E_mon.i[:]

		# Inhibitory population
		s_tpoints_i = n.I_mon.t[:]
		n_inds_i = n.I_mon.i[:]

		s_tpoints = [
			s_tpoints_input_e, 
			s_tpoints_input_i, 
			s_tpoints_e,
			s_tpoints_i
		]

		n_inds = [
			n_inds_input_e,
			n_inds_input_i,
			n_inds_e,
			n_inds_i
		]

		# Pickling
		path_sim = n.path_sim
		sim_id = n.id
		t_run = n.t_run
		exp_type = n.exp_type

		stim_type_e = n.stim_type_e
		stim_size_e = n.stim_size_e
		stim_freq_e = n.stim_freq_e
		stim_offset_e = n.stim_offset_e
		stim_type_i = n.stim_type_i
		stim_size_i = n.stim_size_i
		stim_freq_i = n.stim_freq_i
		stim_pulse_duration = n.stimulus_pulse_duration

		N_input_e = n.N_input_e
		N_input_i = n.N_input_i

		N_e = n.N_e
		N_i = n.N_i

		len_stim_inds_original_E = len(n.stim_inds_original_E)
		w_e_e_max = n.w_e_e_max
		rho_matrix_snapshots_step = n.rho_matrix_snapshots_step

		folder_snaps = os.path.split(n.path_snapshots)[1]
		path_rho = n.path_snapshots
		
		w_matrix_snapshots_step = n.w_matrix_snapshots_step
		path_w = n.path_snapshots

		xpre_matrix_snapshots_step = n.xpre_matrix_snapshots_step
		path_xpre = n.path_snapshots
		xpost_matrix_snapshots_step = n.xpost_matrix_snapshots_step
		
		path_xpost = n.path_snapshots
		plasticity_rule = n.plasticity_rule

		fn = os.path.join(n.path_sim, n.id +'_' + n.exp_type + '.pickle')

		with open(fn, 'wb') as f:
			pickle.dump((
				path_sim, 
				sim_id, 
				nets,
				simulation_flag_pulse_duration,
				t_run, 
				exp_type,
				stim_type_e, 
				stim_type_e,
				stim_size_e,
				stim_freq_e,
				stim_offset_e,
				stim_type_i,
				stim_size_i,
				stim_freq_i,
				stim_pulse_duration,
				pulse_durations,
				N_input_e, 
				N_input_i, 
				N_e, 
				N_i,
				len_stim_inds_original_E,
				w_e_e_max, 
				rho_matrix_snapshots_step,
				folder_snaps,
				w_matrix_snapshots_step, 
				path_w,
				xpre_matrix_snapshots_step, 
				path_xpre,
				xpost_matrix_snapshots_step, 
				path_xpost,
				s_tpoints_input_e, 
				s_tpoints_input_i, 
				s_tpoints_e, 
				s_tpoints_i,
				n_inds_input_e,
				n_inds_input_i,
				n_inds_e, 
				n_inds_i), f)

		print('\nSimulation data pickled to ', fn)

		# Move current simulation folder to superior simulation folder
		shutil.move(n.path_sim, path_sim_folder_superior)

		print('\n > simulation ', i+1, '/', nets)

ext_attractors_plot(
	path_sim_folder_superior, 
	sim_id, 
	exp_type, 
	spiketrains_and_histograms = True,
	rho_matrix_snapshots = False, 
	w_matrix_snapshots = False, 
	xpre_matrix_snapshots = False, 
	xpost_matrix_snapshots = False,
	learning_performance = False, 
	check_wmatrices = False)


