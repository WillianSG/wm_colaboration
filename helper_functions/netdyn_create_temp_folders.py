# -*- coding: utf-8 -*-
"""
@author: slehfeldt

Input:
- path_id: string holding the simulation identifier ('YYYYMMDD_hh_mm_ss')
- add_Ext_att: boolean switch indicating if Ext_att activity is monitored

Output:
- paths: list of strings describing the names of the created folders  

Comments:
- Creates temporary folders into the current stimulation folder for storing simulation data.
"""
import os

def netdyn_create_temp_folders(path_id, add_Ext_att):
	# Temporary folders for simulation data of Input_to_E population
	path_s_tpoints_temp_input_e = path_id + "_s_tpoints_temp_input_e"
	os.mkdir(path_s_tpoints_temp_input_e)

	path_n_inds_temp_input_e = path_id + "_n_inds_temp_input_e"
	os.mkdir(path_n_inds_temp_input_e)

	# Temporary folders for simulation data of Input_to_I population
	path_s_tpoints_temp_input_i = path_id + "_s_tpoints_temp_input_i"
	os.mkdir(path_s_tpoints_temp_input_i)

	path_n_inds_temp_input_i = path_id + "_n_inds_temp_input_i"
	os.mkdir(path_n_inds_temp_input_i)

	# Temporary folders for simulation data of excitatory population
	path_s_tpoints_temp_e = path_id + "_s_tpoints_temp_e"
	os.mkdir(path_s_tpoints_temp_e)

	path_n_inds_temp_e = path_id + "_n_inds_temp_e"
	os.mkdir(path_n_inds_temp_e)

	# Temporary folders for simulation data of inhibitory population
	path_s_tpoints_temp_i = path_id + "_s_tpoints_temp_i"
	os.mkdir(path_s_tpoints_temp_i)

	path_n_inds_temp_i = path_id + "_n_inds_temp_i"
	os.mkdir(path_n_inds_temp_i)

	if add_Ext_att:
		# Temporary folders for simulation data of Ext_att population
		path_s_tpoints_temp_ext_att = path_id + "_s_tpoints_temp_ext_att"
		os.mkdir(path_s_tpoints_temp_ext_att)

		path_n_inds_temp_ext_att = path_id + "_n_inds_temp_ext_att"
		os.mkdir(path_n_inds_temp_ext_att)

		# Temporary folder for weight matrices
		path_wmat_temp_E_att_orig = path_id + "wmat_temp_E_att_orig"
		os.mkdir(path_wmat_temp_E_att_orig)

		path_wmat_temp_E_att_cut = path_id + "wmat_temp_E_cut"
		os.mkdir(path_wmat_temp_E_att_cut)

		path_wmat_temp_Ext_att = path_id + "wmat_temp_Ext_att"


		paths = [path_s_tpoints_temp_input_e, path_n_inds_temp_input_e,path_s_tpoints_temp_input_i, path_n_inds_temp_input_i,path_s_tpoints_temp_e, path_n_inds_temp_e, path_s_tpoints_temp_i, path_n_inds_temp_i, path_s_tpoints_temp_ext_att,path_n_inds_temp_ext_att, path_wmat_temp_E_att_orig,path_wmat_temp_E_att_cut, path_wmat_temp_Ext_att]
	else:
		paths = [path_s_tpoints_temp_input_e, path_n_inds_temp_input_e,path_s_tpoints_temp_input_i, path_n_inds_temp_input_i,path_s_tpoints_temp_e, path_n_inds_temp_e, path_s_tpoints_temp_i, path_n_inds_temp_i]

	return paths