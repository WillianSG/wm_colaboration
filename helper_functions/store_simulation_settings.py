# -*- coding: utf-8 -*-
"""
@author: slehfeldt with adaptations from asonntag

Input:
- network parameters/settings

Output:
- no output arguments

Comments:
- creates a .txt-file before the run of a simulation holding the simulation settings. After creation, the .txt-file is converted to a 'read-only' file. 
"""

import os,stat
import numpy as np

def store_simulation_settings(
	sim_id,
	exp_type,
	iter_count,
	t_run,
	int_meth_neur,
	int_meth_syn,
	dt,
	Input_to_E_mon_record,
	Input_to_I_mon_record,
	E_mon_record,
	I_mon_record,
	Ext_att_record,
	Input_E_rec_record,
	Input_E_rec_attributes_orig,
	Input_I_rec_record,
	Input_I_rec_attributes_orig,
	E_rec_record,
	E_rec_attributes_orig,
	I_rec_record,
	I_rec_attributes_orig,
	E_E_rec_record,
	E_I_rec_record,
	E_I_rec_attributes_orig,
	I_E_rec_record,
	I_E_rec_attributes_orig,
	E_E_rec_attributes_orig,
	Ext_att_E_rec_record,
	Ext_att_E_rec_attributes_orig,
	rec_dt,
	rho_matrix_snapshots,
	rho_matrix_snapshots_step,
	w_matrix_snapshots,
	w_matrix_snapshots_step,
	xpre_matrix_snapshots,
	xpre_matrix_snapshots_step,
	xpost_matrix_snapshots,
	xpost_matrix_snapshots_step,
	stimulus_pulse,
	stimulus_pulse_duration,
	stimulus_pulse_clock_dt,
	stim_type_e,
	stim_size_e,
	stim_freq_e,
	stim_type_i,
	stim_size_i,
	stim_freq_i,
	ext_att_freq,
	plasticity_rule,
	neuron_type,
	N_input_e,
	N_input_i,
	N_e,
	Vr_e,
	Vrst_e,
	Vth_e_init,
	tau_Vth_e,
	Vth_e_incr,
	taum_e,
	tref_e,
	tau_epsp_e,
	tau_ipsp_e,
	N_i,
	Vr_i,
	Vrst_i,
	Vth_i,
	taum_i,
	tref_i,
	tau_epsp_i,
	tau_ipsp_i,
	p_e_e,
	p_e_i,
	p_i_e,
	w_input_e,
	w_input_i,
	w_e_e,
	w_e_i,
	w_i_e,
	w_e_e_max,
	w_ext_att_e,
	add_Ext_att,
	tau_xpre,
	tau_xpost,
	tau_rho,
	thr_b_rho,
	xpre_jump,
	xpost_jump,
	xpre_factor,
	rho_neg,
	eqs_e,
	eqs_i,
	model_E_E,
	pre_E_E,
	post_E_E,
	namespace,
	net):
	# Convert rec_attributes to lists
	Input_E_rec_attributes = []

	for a in np.arange(0,len(Input_E_rec_attributes_orig)):
		Input_E_rec_attributes.append(Input_E_rec_attributes_orig[a])

	Input_I_rec_attributes = []

	for b in np.arange(0,len(Input_I_rec_attributes_orig)):
		Input_I_rec_attributes.append(Input_I_rec_attributes_orig[b])

	E_rec_attributes = []

	for c in np.arange(0,len(E_rec_attributes_orig)):
		E_rec_attributes.append(E_rec_attributes_orig[c])

	I_rec_attributes = []
	
	for d in np.arange(0,len(I_rec_attributes_orig)):
		I_rec_attributes.append(I_rec_attributes_orig[d])

	E_E_rec_attributes = []
	
	for e in np.arange(0,len(E_E_rec_attributes_orig)):
		E_E_rec_attributes.append(E_E_rec_attributes_orig[e])

	E_I_rec_attributes = []

	for f in np.arange(0,len(E_I_rec_attributes_orig)):
		E_I_rec_attributes.append(E_I_rec_attributes_orig[f])

	I_E_rec_attributes = []

	for g in np.arange(0,len(I_E_rec_attributes_orig)):
		I_E_rec_attributes.append(I_E_rec_attributes_orig[g])  

	Ext_att_E_rec_attributes = []

	for h in np.arange(0,len(Ext_att_E_rec_attributes_orig)):
		Ext_att_E_rec_attributes.append(Ext_att_E_rec_attributes_orig[h])

	settings = open(sim_id + "_Simulation_settings_" + str(exp_type) + \
		"_iteration_" + str(iter_count) + ".txt", "w")

	settings.write('###########################################################################\n')

	settings.write('#--- ' + sim_id + ' Simulation settings -------------------------------#\n')

	settings.write('###########################################################################\n\n')
	
	settings.write('#--- Experiment : %s' %(exp_type)+' ----------------------------------------#\n\n')      

	settings.write('###########################################################################\n')
	settings.write('#--- Simulation settings -------------------------------------------------#\n')
	settings.write('###########################################################################\n')
	settings.write('#--- General settings ----------------------------------------------------#\n')
	settings.write('t_run : %s' %(t_run)+'               # Duration of simulation\n')
	settings.write('int_meth_neur : %s' %(int_meth_neur)+'      # Numerical integration method of NeuronGroup\n')
	settings.write('int_meth_syn : %s' %(int_meth_syn)+'        # Numerical integration method of E_E synapses\n')
	settings.write('dt: %s' %(dt)+'                   # Delta t of clock intervals / temporal resolution of simulation \n\n')

	settings.write('#--- Spikemonitors -------------------------------------------------------#\n') 
	settings.write('Input_to_E_mon_record : %s' %(Input_to_E_mon_record)+'\n')
	settings.write('Input_to_I_mon_record : %s' %(Input_to_I_mon_record)+'\n')
	settings.write('E_mon_record: %s' %(E_mon_record)+'\n')
	settings.write('I_mon_record: %s' %(I_mon_record)+'\n')
	settings.write('Ext_att_record: %s' %(Ext_att_record)+'\n\n')

	settings.write('#--- Statemonitors -------------------------------------------------------#\n') 
	settings.write('Input_E_rec_record: %s' %(Input_E_rec_record)+'\n')
	settings.write('Input_E_rec_attributes: %s' %(Input_E_rec_attributes)+'\n')
	settings.write('Input_I_rec_record: %s' %(Input_I_rec_record)+'\n')
	settings.write('Input_I_rec_attributes: %s' %(Input_I_rec_attributes)+'\n')
	settings.write('E_rec_record: %s' %(E_rec_record)+'\n')
	settings.write('E_rec_attributes: %s' %(E_rec_attributes)+'\n')
	settings.write('I_rec_record: %s' %(I_rec_record)+'\n')
	settings.write('I_rec_attributes: %s' %(I_rec_attributes)+'\n')
	settings.write('E_E_rec_record: %s' %(E_E_rec_record)+'\n')
	settings.write('E_E_rec_attributes: %s' %(E_E_rec_attributes)+'\n')
	settings.write('E_I_rec_record: %s' %(E_I_rec_record)+'\n')
	settings.write('E_I_rec_attributes: %s' %(E_I_rec_attributes)+'\n')
	settings.write('I_E_rec_record: %s' %(I_E_rec_record)+'\n')
	settings.write('I_E_rec_attributes: %s' %(I_E_rec_attributes)+'\n')
	settings.write('Ext_att_E_rec_record: %s' %(Ext_att_E_rec_record)+'\n')
	settings.write('Ext_att_E_rec_attributes: %s' %(Ext_att_E_rec_attributes)+'\n')
	settings.write('rec_dt: %s' %(rec_dt)+' # Delta t of state variable recording \n\n')

	settings.write('#--- @network_operation settings -----------------------------------------#\n')
	settings.write('# Rho snapshots \n')                
	settings.write('rho_matrix_snapshots: %s' %(rho_matrix_snapshots)+'\n') 
	settings.write('rho_matrix_snapshots_step: %s' %(rho_matrix_snapshots_step)+'\n\n') 

	settings.write('# Weight snapshots \n')                
	settings.write('w_matrix_snapshots: %s' %(w_matrix_snapshots)+'\n') 
	settings.write('w_matrix_snapshots_step: %s' %(w_matrix_snapshots_step)+'\n\n') 

	settings.write('# Xpre snapshots \n') 
	settings.write('xpre_matrix_snapshots: %s' %(xpre_matrix_snapshots)+'\n') 
	settings.write('xpre_matrix_snapshots_step: %s' %(xpre_matrix_snapshots_step)+'\n\n') 

	settings.write('# Xpost snapshots \n') 
	settings.write('xpost_matrix_snapshots: %s' %(xpost_matrix_snapshots)+'\n') 
	settings.write('xpost_matrix_snapshots_step: %s' %(xpost_matrix_snapshots_step)+'\n\n') 

	settings.write('# Switches of stimulation protocols for testing attractors\n')   
	settings.write('# Stimulus pulse\n')
	settings.write('stimulus_pulse: %s' %(stimulus_pulse)+'\n')  
	settings.write('stimulus_pulse_duration: %s' %(stimulus_pulse_duration)+'\n') 
	settings.write('stimulus_pulse_clock_dt: %s' %(stimulus_pulse_clock_dt)+'\n\n')


	settings.write('###########################################################################\n')
	settings.write('#--- Stimulus settings ---------------------------------------------------#\n')
	settings.write('###########################################################################\n')
	settings.write('#--- Stimulus to E -------------------------------------------------------#\n')
	settings.write('stim_type_e : %s' %(stim_type_e)+'     \n')
	settings.write('stim_size_e : %s' %(stim_size_e)+'     \n')
	settings.write('stim_freq_e : %s' %(stim_freq_e)+'    \n')

	settings.write('#--- Stimulus to I -------------------------------------------------------#\n')
	settings.write('stim_type_i : %s' %(stim_type_i)+'     \n')
	settings.write('stim_size_i : %s' %(stim_size_i)+'     \n')
	settings.write('stim_freq_i : %s' %(stim_freq_i)+'    # Stimulus frequency to I \n')

	settings.write('#--- Activation frequency of external attractor --------------------------#\n')
	settings.write('ext_att_freq : %s' %(ext_att_freq)+' \n\n')

	settings.write('###########################################################################\n')
	settings.write('#--- Learning and noise settings -----------------------------------------#\n')
	settings.write('###########################################################################\n')          
	settings.write('#--- Plasticity rule: calcium_original, calcium_adjusted, none -----------#\n')               
	settings.write('plasticity_rule : %s' %(plasticity_rule)+' \n\n')

	settings.write('###########################################################################\n')
	settings.write('#--- Neuron parameters ---------------------------------------------------#\n')
	settings.write('###########################################################################\n')              
	settings.write('#--- Neuron type: LIF, poisson, spikegenerators --------------------------#\n')               
	settings.write('neuron_type : %s' %(neuron_type)+'      \n\n')

	settings.write('#--- Optional external population (copy of E_E wit fixed attractor) ------#\n')
	settings.write('add_Ext_att : %s' %(add_Ext_att)+'  # If True: add Ext_att population for testing of ETF functionn\n\n')  

	settings.write('#--- Input population ----------------------------------------------------#\n')      
	settings.write('N_input_e : %s' %(N_input_e)+'        # Number of input neurons to E \n')
	settings.write('N_input_i : %s' %(N_input_i)+'         # Number of input neurons to I \n\n')

	settings.write('#--- Excitatory population -----------------------------------------------#\n')
	settings.write('N_e : %s' %(N_e)+'              # Number of excitatory neurons\n')
	settings.write('Vr_e : %s' %(Vr_e)+'         # Resting potential of excitatory neruons\n')
	settings.write('Vrst_e : %s' %(Vrst_e)+'       # Re  set potential of excitatory neruons\n')
	settings.write('Vth_e_init : %s' %(Vth_e_init)+'   # Initial threshold voltage of excitatory neruons\n')
	settings.write('Vth_e_incr : %s' %(Vth_e_incr)+'     # Post-spike threshold voltage increase\n')                
	settings.write('tau_Vth_e : %s' %(tau_Vth_e)+'     # Time constant of threshold decay\n')
	settings.write('taum_e : %s' %(taum_e)+'        # Membrane time constant of excitatory neruons\n')
	settings.write('tref_e : %s' %(tref_e)+'         # Refractory period of excitatory neruons\n')
	settings.write('tau_epsp_e : %s' %(tau_epsp_e)+'    # Time constant of EPSP in excitatory neruons\n')
	settings.write('tau_ipsp_e : %s' %(tau_ipsp_e)+'    # Time constant of IPSP in excitatory neruons\n\n')

	settings.write('#--- Inhibitory population -----------------------------------------------#\n')
	settings.write('N_i : %s' %(N_i)+'                # Number of inhibitory neurons\n')
	settings.write('Vr_i : %s' %(Vr_i)+'          # Resting voltage of inhibitory neurons\n')
	settings.write('Vrst_i : %s' %(Vrst_i)+'        # Re  set potential of inhibitory neurons\n')
	settings.write('Vth_i : %s' %(Vth_i)+'         # Threshold voltage of inhibitory neurons\n')
	settings.write('taum_i : %s' %(taum_i)+'         # Membrane time constant of inhibitory neurons\n')
	settings.write('tref_i : %s' %(tref_i)+'          # Refractory period of inhibitory neurons\n')
	settings.write('tau_epsp_i : %s' %(tau_epsp_i)+'     # Time constant of EPSP in inhibitory neurons\n')
	settings.write('tau_ipsp_i : %s' %(tau_ipsp_i)+'     # Time constant of IPSP in inhibitory neurons\n\n')

	settings.write('###########################################################################\n')
	settings.write('#--- Synapse parameters --------------------------------------------------#\n')
	settings.write('###########################################################################\n')
	settings.write('#--- Connection probabilities --------------------------------------------#\n')
	settings.write('p_e_e : %s' %(p_e_e)+'              # Probability excitatory to excitatory\n')
	settings.write('p_e_i : %s' %(p_e_i)+'             # Probability excitatory to inhibitory\n')
	settings.write('p_i_e : %s' %(p_i_e)+'             # Probability inhibitory to excitatory\n\n')

	settings.write('#--- Connection weights --------------------------------------------------#\n')
	settings.write('w_input_e : %s' %(w_input_e)+'        # Weight input to excitatory\n')
	settings.write('w_input_i : %s' %(w_input_i)+'        # Weight input to inhibitory\n')
	settings.write('w_e_e : %s' %(w_e_e)+'            # Weight excitatory to excitatory\n')
	settings.write('w_e_i : %s' %(w_e_i)+'            # Weight excitatory to inhibitory\n')
	settings.write('w_i_e : %s' %(w_i_e)+'            # Weight inhibitory to excitatory\n\n')

	settings.write('#--- Other connection weight variables -----------------------------------#\n')               
	settings.write('w_e_e_max : %s' %(w_e_e_max)+'           # Upper synaptic weight boundary for E_E connections\n')
	settings.write('w_ext_att_e : %s' %(w_ext_att_e)+'            # Weight external attractor to attractor in E \n\n')

	settings.write('###########################################################################\n')
	settings.write('#--- New learning rule parameters ------------------------------------#\n')
	settings.write('###########################################################################\n')
	settings.write('#--- X parameters --------------------------------------------------# \n')

	settings.write('#--- Synaptic efficacy parameters ----------------------------------------#\n')               
	settings.write('tau_rho : %s' %(tau_rho)+'    # Time constant of synaptic efficacy (rho) changes \n')
	settings.write('thr_b_rho : %s' %(thr_b_rho)+'  # Threshold of bistabiltiy of synaptic efficacy \n\n')

	settings.write('#--- Parameters for adjusted calcium-based rule --------------------------# \n')
	settings.write('xpre_jump : %s' %(xpre_jump)+'        # X_pre Jumpsize\n') 
	settings.write('xpost_jump : %s' %(xpost_jump)+'      # X_post Jumpsize\n')
	settings.write('tau_xpre: %s' %(tau_xpre)+'           # Time constant of x_pre variable changes \n')
	settings.write('tau_xpost: %s' %(tau_xpost)+'         # Time constant of x_post variable changes \n')
	settings.write('tau_xpre: %s' %(xpre_factor)+'        # Scaling factor for xpre variable that adds to efficacy \n')
	settings.write('tau_xpost: %s' %(rho_neg)+'          # fixed negative synaptic efficacy (rho) change variable \n')

	settings.write('###########################################################################\n')
	settings.write('#--- Neurons: Neuron model equations -------------------------------------#\n')
	settings.write('###########################################################################\n\n')
	settings.write('#--- Excitatory neurons: LIF with adaptive threshold ---------------------#\n')
	settings.write('eqs_e : %s' %(eqs_e)+'\n\n')
	settings.write('#--- Inhibitory neurons: LIF with constant threshold ---------------------#\n')
	settings.write('eqs_i : %s' %(eqs_i)+'\n\n')

	settings.write('###########################################################################\n')
	settings.write('#--- Synapses: E_E synapse model equations -------------------------------#\n')
	settings.write('###########################################################################\n\n')              
	settings.write('model_E_E : %s' %(model_E_E)+'\n\n')
	settings.write('pre_E_E : %s' %(pre_E_E)+'\n\n')
	settings.write('post_E_E : %s' %(post_E_E)+'\n\n')

	settings.write('###########################################################################\n')
	settings.write('#--- Namespace -----------------------------------------------------------#\n')
	settings.write('###########################################################################\n\n') 
	settings.write('namespace : %s' %(namespace)+'\n\n')

	settings.write('###########################################################################\n')
	settings.write('#--- Network -------------------------------------------------------------#\n')
	settings.write('###########################################################################\n\n') 
	settings.write('net : %s' %(net)+'\n\n')
	settings.close()

	#      Change mode of .txt-file to read-only 
	os.chmod(sim_id+"_Simulation_settings_"+str(exp_type)+"_iteration_"+str(iter_count)+".txt", stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)

