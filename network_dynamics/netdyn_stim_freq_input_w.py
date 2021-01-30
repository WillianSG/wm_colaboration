# -*- coding: utf-8 -*-
"""
@author: slehfeldt with adaptations by asonntag

Input:

Output:

Comments:
- Network dynamics with changing stimulus frequency and input weight.
"""
from brian2 import *
import os, sys, shutil, pickle, socket
from time import localtime
import shutil
import warnings

# Ignore warning if stored list in dictionary is empty
warnings.filterwarnings('ignore')

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
from netdyn_stimfreq_inputweight_makeplots import *

# 1 - Simulation settings

# String describing the type of experiment
exp_type = 'netdyn_stimfreq_inputweight_' #'_E_only'
simulation_flag = 'I_only' # Options: E_only and I_only 

# Input frequency range

f_start = 4700
f_end = 10000
f_step = 300
f_range = np.arange(f_start, f_end, f_step)*Hz

# Weight range

w_start = 1
w_end = 1.1
w_step = 1
w_range = np.arange(w_start, w_end, w_step)*mV

n = AttractorNetwork() # network class
n.exp_type = exp_type + simulation_flag

# 1.1 - Simulation results settings

# Create folders for temporary storage of simulation data
n.init_simulation_folder()
sim_paths_temp = netdyn_create_temp_folders(path_id = n.path_sim_id,
add_Ext_att = False)
sim_path_folder = n.path_sim_id + '_' + n.exp_type 

# General settings

n.t_run = 1*second # Duration of simulation  

n.int_meth_neur = 'linear' # Numerical integration method of NeuronGroup
n.int_meth_syn = 'linear'    # Numerical integration method of Synapses

n.dt = 0.1*ms # Delta t of clock intervals / temporal resolution of simulation

# Monitor settings (Spikemonitors)

n.Input_to_E_mon_record = True
n.Input_to_I_mon_record = True
n.E_mon_record = True
n.I_mon_record = True

# 1.2 - Stimulus settings

n.stim_type_e = 'flat_to_E_fixed_size'
n.stim_size_e = 64
n.stim_freq_e = 0*Hz 
n.stim_offset_e = 96
n.stim_deg_type_e = 'no_stim_deg'

n.stim_type_i = 'flat_to_I'
n.stim_size_i = n.N_i
n.stim_freq_i = 0*Hz

# 1.3 - Learning rule settings

n.plasticity_rule = 'none' # Learning rule
n.neuron_type = 'LIF'
n.bistability = False  
n.parameter = '2.1'

# 1.4 - Initialise and store a network with all connetions = 0 mV. "For experiments with a subset of weights fixed, specify here as well."

# Connection weights
n.w_input_e = 0*mV      # Weight input to excitatory
n.w_input_i = 0*mV      # Weight input to inhibitory
n.w_e_e = 0*mV          # Weight excitatory to excitatory
n.w_e_i = 0*mV          # Weight excitatory to inhibitory
n.w_i_e = 0*mV          # Weight inhibitory to excitatory    
n.fixed_attractor = False

# Initialise net and store
n.init_network_modules()
n.net.store()

# 2 - Simulation

count = 0

# Loop through defined range frequencies and weights
for freq in f_range:
	for weight in w_range:
		# Restore initial network configuration        
		n.net.restore()

		# E_only: Initialise stimulus and input weight
		if simulation_flag == 'E_only':    
			n.stim_freq_e = freq
			n.init_stimulus_e()

			n.w_input_e = weight
			n.init_weights()

		# I_only: Initialise stimulus and input weight  
		if simulation_flag == 'I_only':  
			n.stim_freq_i = freq
			n.init_stimulus_i()

			n.w_input_i = weight
			n.init_weights()

		# Run simulation
		n.run_net(report = None)

		# Simulation data: a) timepoints of spikes (s_tpoints) and neuron indices (n_inds)

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
		netdyn_store_temp_data(
			var_outer = freq/Hz, 
			var_inner = weight/mV, 
			folder = sim_path_folder, 
			paths_temp = sim_paths_temp, 
			data_temp = sim_data_temp)

		# Command line output during simulation

		count += 1

		if simulation_flag == 'E_only': 
			print( '\nStimulus frequency: ', n.stim_freq_e)
			print( 'Input weight: ', n.w_input_e)
			print( 'Finished iteration ', count, '/', 
				len(f_range)*len(w_range))

		if simulation_flag == 'I_only':
			print( '\nStimulus frequency: ', n.stim_freq_i)
			print( 'Input weight: ', n.w_input_i)
			print( 'Finished iteration ', count, '/', 
				len(f_range)*len(w_range))

# After simulation: move stored data to dictionaries      
[s_tpoints_input_e, 
n_inds_input_e, 
s_tpoints_input_i, 
n_inds_input_i, 
s_tpoints_e, 
n_inds_e, 
s_tpoints_i, 
n_inds_i] = netdyn_store_as_dict(
	range_outer = f_range, 
	range_inner = w_range, 
	cwd = n.cwd, 
	folder = sim_path_folder, 
	paths_temp = sim_paths_temp, 
	add_Ext_att = n.add_Ext_att)

# 3 - Store recorded data as .pickle file

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
	f_range,
	f_step, 
	w_range, 
	w_step, 
	s_tpoints_input_e, 
	s_tpoints_input_i, 
	s_tpoints_e, 
	s_tpoints_i,
	n_inds_input_e, 
	n_inds_input_i, 
	n_inds_e, 
	n_inds_i), f)

print('\n > Simulation data pickled to ', fn)

os.chdir(n.path_sim)

# Delete temporary folders
for d in range(0,len(sim_paths_temp)):   
	shutil.rmtree(sim_paths_temp[d]) 

netdyn_stimfreq_inputweight_makeplots(path_sim, sim_id, exp_type)

