# -*- coding: utf-8 -*-
"""
@author: slehfeldt

Inputs:
- s_tpoints_xy: array with time points of spikes
- n_inds_xy: array with neuron indices belonging to tpoints
- t_run: duration of simulation
- stim_pulse_duration: duration of stimulus
- size_attractor: size of the expected attractor / stimulus size
- plot_spiketrains: flag to decide whether spiking data on which classification
  is based should be plotted
- sim_id: simulation id needed for plotting
- path_sim: path to simulation folder needed for plotting

Outputs:
- delay_activities: list with True and False
  [False, False]: no delay activity
  [True, False]: fading delay activity
  [True, True]: delay activity

Comments:
- Classifies a recorded spike train to 'No delay activity' (delay_activities = [False, False]), 'Fading delay activity' (delay_activities = [True, False]) and 'Delay activity' (delay_activities = [True, True]). 
- The function checks for a user-defined critical number of active neuron indices in a user-defined temporal window after simulus offset and in a same temporal window before the end of the simulation.
"""

def learning_check_for_delay_activity(s_tpoints_input_e, n_inds_input_e, s_tpoints_e, n_inds_e, t_run, stim_pulse_duration, size_attractor, plot_spiketrains, sim_id, path_sim):
	import sys
	from brian2 import second

	from extract_trial_data import *
	from find_wmax_for_attractors_plot_spiketrains import *

	# User-defined critical number of active indices within the observed period
	critical_num_inds_active = size_attractor / 2

	period_duration = 0.1*second
	delay_activities = {}

	delay_activity_p1 = False
	delay_activity_p2 = False    

	# === Period 1: check after stimulus offset

	# Define start and end time point  
	t_start_p1 = stim_pulse_duration
	t_end_p1 = stim_pulse_duration + period_duration

	# Extract trial data in temporal window
	[inds_temp_p1, tpoints_temp_p1] = extract_trial_data(t_start = t_start_p1,t_end = t_end_p1,	inds = n_inds_e, tpoints = s_tpoints_e)

	# Remove dublicates in inds_temp
	inds_temp_no_dublicates_p1 = set(inds_temp_p1)

	# Check for delay activity
	if len(inds_temp_no_dublicates_p1) > critical_num_inds_active:
		delay_activity_p1 = True 

	# === Period 2: check before end of simulation 

	# Define start and end time point      
	t_start_p2 = t_run - period_duration 
	t_end_p2 = t_run 

	# Extract trial data in temporal window  
	[inds_temp_p2, tpoints_temp_p2] = extract_trial_data(t_start = t_start_p2, t_end = t_end_p2, inds = n_inds_e, tpoints = s_tpoints_e)

	# Remove dublicates in inds_temp
	inds_temp_no_dublicates_p2 = set(inds_temp_p2)

	# Check for delay activity  
	if len(inds_temp_no_dublicates_p2) > critical_num_inds_active:
		delay_activity_p2 = True

	# === Plot spiketrains
	if plot_spiketrains:
		find_wmax_for_attractors_plot_spiketrains(
			sim_id = sim_id, 
			path_sim = path_sim, 
			s_tpoints_input_e = s_tpoints_input_e, 
			n_inds_input_e = n_inds_input_e,
			s_tpoints_e = s_tpoints_e, 
			n_inds_e = n_inds_e,
			tpoints_temp_p1 = tpoints_temp_p1,
			inds_temp_p1 = inds_temp_p1,
			tpoints_temp_p2 = tpoints_temp_p2,
			inds_temp_p2 = inds_temp_p2,
			t_run = t_run,
			t_start = [t_start_p1, t_start_p2],
			t_end = [t_end_p1, t_end_p2],
			stim_pulse_duration = stim_pulse_duration)

	# Classification results
	delay_activities = [delay_activity_p1, delay_activity_p2]

	return delay_activities 