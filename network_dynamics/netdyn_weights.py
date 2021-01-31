# -*- coding: utf-8 -*-
"""
@author: wgirao adapted from asonntag

Input:

Output:

Comments:
- Network dynamics with changing stimulus frequency and input weight.
- The netdyn_* scripts are in regard to the network dynamic for setting up the general network activity.
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
from netdyn_create_temp_folders import *
from netdyn_store_temp_data import *
from netdyn_store_as_dict import *
from netdyn_weights_makeplots import *

# 1 - Experiment results folder

exp_name = 'netdyn_weight_wee' # [?] - "_wee"? (weight E to E?)

simulation_folder = os.path.join(parent_dir, 'net_simulation')

if not(os.path.isdir(simulation_folder)):
	os.mkdir(simulation_folder)

idt = localtime()  

simulation_id = str(idt.tm_year) + '{:02}'.format(idt.tm_mon) + \
	'{:02}'.format(idt.tm_mday) + '_'+ '{:02}'.format(idt.tm_hour) + '_' + \
	'{:02}'.format(idt.tm_min) + '_' + '{:02}'.format(idt.tm_sec)

sim_results_folder = os.path.join(simulation_folder, simulation_id + exp_name)

if not(os.path.isdir(sim_results_folder)):
	os.mkdir(sim_results_folder)

# 2 - Simulation settings

simulation_flag = 'weight_wee' # 'weight_wee', 'weights_wei_wie'

# Weight range 1
w1_start = 0
w1_end = 2.1
w1_step = 0.1
w1_range = np.arange(w1_start, w1_end, w1_step)*mV

# Weight range 2
w2_start = 0
w2_end = 1
w2_step = 1
w2_range = np.arange(w2_start, w2_end, w2_step)*mV

# 2.1 - Network settings

n = AttractorNetwork() # network class

# State
n.exp_type = exp_name
n.t_run = 1*second # Duration of simulation  
n.int_meth_neur = 'linear' # Numerical integration method of NeuronGroup
n.int_meth_syn = 'linear' # Numerical integration method of Synapses
n.dt = 0.1*ms # Temporal resolution of simulation

# Monitors
n.Input_to_E_mon_record = True
n.Input_to_I_mon_record = True
n.E_mon_record = True
n.I_mon_record = True

# Stimulus
n.stim_type_e = 'flat_to_E_fixed_size'
n.stim_size_e = 64
n.stim_freq_e = 3900*Hz 
n.stim_offset_e = 96
n.stim_deg_type_e = 'no_stim_deg'

n.stim_type_i = 'flat_to_I'
n.stim_size_i = n.N_i
n.stim_freq_i = 5000*Hz

# Learning rule
n.plasticity_rule = 'LR2'
n.parameter = '2.1'
n.neuron_type = 'LIF' 
n.bistability = False

# Connection weights
n.w_input_e = 1*mV      # Weight input to excitatory
n.w_input_i = 1*mV      # Weight input to inhibitory
n.w_e_e = 0*mV          # Weight excitatory to excitatory
n.w_e_i = 1*mV          # Weight excitatory to inhibitory
n.w_i_e = 1*mV          # Weight inhibitory to excitatory

n.fixed_attractor = False

# Initialise net and store
n.init_network_modules()
n.net.store()

# Create folders for temporary storage of simulation data
n.set_simulation_folder()

sim_paths_temp = netdyn_create_temp_folders(path_id = n.path_sim_id, add_Ext_att = False)

sim_path_folder = n.path_sim_id + '_' + n.exp_type

count = 0

# Loop through defined range frequencies and weights
for w1 in w1_range:
	for w2 in w2_range:
		# Restore initial network configuration        
		n.net.restore()

		# Initialise weights

		if simulation_flag == 'weight_wee':
			n.w_e_e = w1
			n.set_weights()

		if simulation_flag == 'weights_wei_wie':
			n.w_e_i = w1
			n.w_i_e = w2
			n.set_weights()

		# Run simulation
		n.run_net(report = None)

		# Simulation data
		# Timepoints of spikes (s_tpoints) and neuron indices (n_inds)

		# Input_to_E population 
		s_tpoints_temp_input_e = n.Input_to_E_mon.t
		n_inds_temp_input_e = n.Input_to_E_mon.i

		# Input_to_I population 
		s_tpoints_temp_input_i = n.Input_to_I_mon.t
		n_inds_temp_input_i = n.Input_to_I_mon.i

		# Excitatory population
		s_tpoints_temp_e = n.E_mon.t
		n_inds_temp_e = n.E_mon.i

		# Inhibitory population
		s_tpoints_temp_i = n.I_mon.t
		n_inds_temp_i = n.I_mon.i

		sim_data_temp = [s_tpoints_temp_input_e, n_inds_temp_input_e,s_tpoints_temp_input_i, n_inds_temp_input_i, s_tpoints_temp_e, n_inds_temp_e, s_tpoints_temp_i, n_inds_temp_i]

		# Store simulation data into temporary folder
		netdyn_store_temp_data(var_outer = w1/mV, var_inner = w2/mV,
			folder = sim_path_folder, 
			paths_temp = sim_paths_temp, 
			data_temp = sim_data_temp)


		count += 1

		if simulation_flag == 'weight_wee': 
			print('\nE to E weight: ', n.w_e_e)
			print('Finished iteration ', count, '/', len(w1_range)*len(w2_range))

		if simulation_flag == 'weights_wei_wie':
			print('\nE to I weight: ', n.w_e_i)
			print('I to E weight: ', n.w_i_e)
			print('Finished iteration ', count, '/', len(w1_range)*len(w2_range))

# After simulation, move stored data to dictionaries      
[s_tpoints_input_e, n_inds_input_e, s_tpoints_input_i, n_inds_input_i,s_tpoints_e, n_inds_e, s_tpoints_i, n_inds_i] = netdyn_store_as_dict(
	range_outer = w1_range, 
	range_inner = w2_range, 
	cwd = n.cwd, 
	folder = sim_path_folder, 
	paths_temp = sim_paths_temp, 
	add_Ext_att = n.add_Ext_att)

# Store recorded data as .pickle file
path_sim = n.path_sim
sim_id = n.id
t_run = n.t_run
exp_type = n.exp_type

stim_type_e = n.stim_type_e
stim_size_e = n.stim_size_e
stim_freq_e = n.stim_freq_e
stim_offset_e = n.stim_offset_e
stim_deg_type_e = n.stim_deg_type_e
stim_type_i = n.stim_type_i
stim_size_i = n.stim_size_i
stim_freq_i = n.stim_freq_i

N_input_e = n.N_input_e
N_input_i = n.N_input_i
N_e = n.N_e
N_i = n.N_i

fn = os.path.join(path_sim, n.id + '_' + n.exp_type + '.pickle')

with open(fn,'wb') as f:
	pickle.dump((
		path_sim, 
		sim_id, 
		t_run, 
		exp_type,
		simulation_flag,
		stim_type_e,
		stim_size_e,
		stim_freq_e,
		stim_offset_e,
		stim_deg_type_e,
		stim_type_i,
		stim_size_i,
		stim_freq_i,
		N_input_e, 
		N_input_i, 
		N_e, 
		N_i,
		w1_range,
		w1_step, 
		w2_range, 
		w2_step, 
		s_tpoints_input_e, 
		s_tpoints_input_i, 
		s_tpoints_e, 
		s_tpoints_i,
		n_inds_input_e, 
		n_inds_input_i, 
		n_inds_e, 
		n_inds_i),f)

print('\nSimulation data pickled to ', fn)

os.chdir(n.path_sim)

# Delete temporary folders
for d in range(0, len(sim_paths_temp)):   
	shutil.rmtree(sim_paths_temp[d])

netdyn_weights_makeplots(path_sim, sim_id, exp_type)

print('\nnetdyn_weights.py - END.\n')